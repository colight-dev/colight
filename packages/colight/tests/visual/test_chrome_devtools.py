"""Tests for Chrome DevTools isolation, tab lifecycle and screenshots.

Every launched Chrome owns an ephemeral debug port and a temporary profile
directory, so tab-count assertions here are per-instance truths — no other
process (or test session) can add or remove tabs behind our back.
"""

import concurrent.futures
import json
import urllib.request

import pytest

from colight.chrome_devtools import ChromeContext, shutdown_chrome


def get_open_tabs(port):
    """List page tabs on a Chrome instance, excluding the startup data: tab."""
    try:
        response = urllib.request.urlopen(f"http://localhost:{port}/json", timeout=2)
        tabs = json.loads(response.read())
        return [
            tab
            for tab in tabs
            if tab.get("type") == "page" and tab.get("url") != "data:,"
        ]
    except Exception:
        return []


def assert_tab_count(port, expected):
    tabs = get_open_tabs(port)
    assert len(tabs) == expected, (
        f"expected {expected} tab(s) on port {port}, got {len(tabs)}: "
        f"{[t.get('title') for t in tabs]}"
    )


@pytest.fixture(autouse=True)
def _clean_chrome():
    shutdown_chrome()
    yield
    shutdown_chrome()


TEST_HTML = """
<html>
<head><title>Chrome DevTools Test</title></head>
<body style="background: #4ecdc4; margin: 0;">
    <h1>Chrome DevTools Test</h1>
</body>
</html>
"""


def test_basic_lifecycle_and_screenshot(tmp_path):
    """Startup, tab creation, HTML loading, screenshot capture, cleanup."""
    with ChromeContext(width=800, height=600, keep_alive=0) as chrome:
        assert chrome.port is not None
        assert_tab_count(chrome.port, 1)

        chrome.load_html(TEST_HTML)
        image_data = chrome.capture_image()
        assert image_data[:8] == b"\x89PNG\r\n\x1a\n"
        (tmp_path / "shot.png").write_bytes(image_data)

    # keep_alive=0: instance is torn down with the last context.
    assert get_open_tabs(chrome.port) == []
    with pytest.raises(Exception):
        urllib.request.urlopen(
            f"http://localhost:{chrome.port}/json/version", timeout=1
        )


def test_isolated_instance_uses_temp_profile_and_cleans_up():
    ctx = ChromeContext(width=300, height=200, reuse=False)
    ctx.start()
    try:
        instance = ctx._instance
        assert instance is not None
        assert instance.user_data_dir.exists()
        assert (instance.user_data_dir / "DevToolsActivePort").exists()
    finally:
        ctx.stop()
    assert instance.process.poll() is not None
    assert not instance.user_data_dir.exists()


def test_multiple_contexts_share_one_instance():
    """Contexts in one process share a Chrome; each owns exactly one tab."""
    contexts = []
    for i in range(3):
        ctx = ChromeContext(width=300, height=200, keep_alive=0)
        ctx.start()
        ctx.load_html(f"<html><body>Context {i + 1}</body></html>")
        contexts.append(ctx)

    ports = {ctx.port for ctx in contexts}
    assert len(ports) == 1  # shared instance
    assert_tab_count(contexts[0].port, 3)

    contexts[0].stop()
    assert_tab_count(contexts[0].port, 2)
    contexts[1].stop()
    assert_tab_count(contexts[0].port, 1)
    contexts[2].stop()  # keep_alive=0: last context tears the instance down
    assert get_open_tabs(contexts[0].port) == []


def test_concurrent_isolated_instances_do_not_interfere():
    """Two private Chrome instances run concurrently without collisions."""

    def run(color: str):
        with ChromeContext(width=200, height=150, reuse=False) as ctx:
            ctx.load_html(f"<html><body style='background:{color};'></body></html>")
            tabs = get_open_tabs(ctx.port)
            image = ctx.capture_image()
            body = ctx.evaluate("document.body.style.background")
            return ctx.port, len(tabs), body, image

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        red = pool.submit(run, "red")
        blue = pool.submit(run, "blue")
        port_a, tabs_a, body_a, image_a = red.result(timeout=60)
        port_b, tabs_b, body_b, image_b = blue.result(timeout=60)

    assert port_a != port_b  # ephemeral ports, no shared 9222
    # Each instance sees exactly its own tab: per-instance truth.
    assert tabs_a == 1
    assert tabs_b == 1
    assert body_a == "red"
    assert body_b == "blue"
    assert image_a != image_b


def test_explicit_attach_to_existing_instance():
    """Opt-in reuse: port= attaches to a running Chrome and never kills it."""
    owner = ChromeContext(width=300, height=200, reuse=False)
    owner.start()
    try:
        owner.load_html("<html><body>owner</body></html>")
        attached = ChromeContext(width=300, height=200, port=owner.port)
        attached.start()
        try:
            attached.load_html("<html><body>attached</body></html>")
            assert attached.port == owner.port
            assert_tab_count(owner.port, 2)
        finally:
            attached.stop()
        # Detaching closes only the attached tab; the browser survives.
        assert_tab_count(owner.port, 1)
        assert owner._instance is not None and owner._instance.is_alive()
    finally:
        owner.stop()


def test_attach_to_missing_instance_fails_fast():
    ctx = ChromeContext(port=1)  # nothing listens on port 1
    with pytest.raises(RuntimeError, match="No Chrome DevTools endpoint"):
        ctx.start()
    ctx.server.stop()


def test_shutdown_chrome_closes_everything():
    ctx = ChromeContext(width=300, height=200, keep_alive=30)
    ctx.start()
    port = ctx.port
    ctx.stop()  # keep_alive would hold the instance for 30s
    shutdown_chrome()
    with pytest.raises(Exception):
        urllib.request.urlopen(f"http://localhost:{port}/json/version", timeout=1)


def test_webgpu_support_detection():
    with ChromeContext(width=400, height=300, keep_alive=0) as chrome:
        webgpu_info = chrome.check_webgpu_support()
        assert isinstance(webgpu_info, dict)
        assert "supported" in webgpu_info


def test_tab_cleanup_stress():
    """Create many contexts rapidly; every tab is cleaned up."""
    num_contexts = 10
    contexts = []
    for i in range(num_contexts):
        ctx = ChromeContext(width=200, height=150, keep_alive=0)
        ctx.start()
        ctx.load_html(f"<html><body>Stress Test Context {i + 1}</body></html>")
        contexts.append(ctx)

    assert_tab_count(contexts[0].port, num_contexts)

    for ctx in contexts:
        ctx.stop()

    assert get_open_tabs(contexts[0].port) == []
