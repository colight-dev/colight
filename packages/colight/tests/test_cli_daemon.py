"""Tests for the colight daemon: lifecycle, discovery, routing, warm scenes.

Discovery/lifecycle/auth tests run everywhere (the daemon launches Chrome
lazily, so a Chrome-less environment can still bind the HTTP service).
Render-path tests drive the real CLI through a live daemon and are skipped
when Chrome or the JS bundle is missing, same mechanism as the other
visual tests.
"""

import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from typing import Iterator, Tuple

import pytest
from click.testing import CliRunner

import colight.env as env
from colight.chrome_devtools import find_chrome
from colight.cli_tools import daemon as daemon_mod
from colight.cli_tools import daemon_client
from colight_cli import main as cli_main

SCENE = """import colight.scene3d as S

scene = (
    S.Ellipsoid(
        centers=[[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        colors=[[1, 0, 0], [0, 1, 0]],
    )
    + {"defaultCamera": {"position": [0, 0, 12], "target": [0, 0, 0],
                         "up": [0, 1, 0], "fov": 45}}
)
scene
"""

SIZE_ARGS = ["--width", "400", "--height", "400"]


def _require_renderer() -> None:
    widget_path = env.WIDGET_PATH
    if not (isinstance(widget_path, pathlib.Path) and widget_path.exists()):
        pytest.skip("colight JS bundle not built (js-dist missing)")
    try:
        chrome_path = find_chrome()
    except FileNotFoundError:
        chrome_path = None
    if not chrome_path:
        pytest.skip("Chrome not found for daemon tests")


@pytest.fixture
def project(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    return tmp_path


@pytest.fixture
def server(project: pathlib.Path) -> Iterator[daemon_mod.DaemonServer]:
    daemon = daemon_mod.DaemonServer(project, idle_timeout=300.0, pool_size=2)
    daemon.start()
    try:
        yield daemon
    finally:
        daemon.shutdown()


def _info(server: daemon_mod.DaemonServer) -> daemon_client.DaemonInfo:
    info = daemon_client.read_daemon_file(server.file_path)
    assert info is not None
    return info


def _status(server: daemon_mod.DaemonServer) -> dict:
    return daemon_client.request(_info(server), "GET", "/status")


class TestDiscovery:
    def test_no_file_returns_none(self, project: pathlib.Path):
        assert daemon_client.discover(project) is None

    def test_walks_up_from_nested_dir(
        self, project: pathlib.Path, server: daemon_mod.DaemonServer
    ):
        nested = project / "docs" / "examples"
        nested.mkdir(parents=True)
        found = daemon_client.discover(nested)
        assert found is not None
        assert found.port == server.port

    def test_dead_pid_removes_stale_file(self, project: pathlib.Path):
        # A reaped child pid is a realistic dead pid.
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        path = daemon_client.daemon_file_path(project)
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "port": 1,
                    "pid": child.pid,
                    "version": daemon_client.colight_version(),
                    "token": "t",
                    "started": time.time(),
                }
            )
        )
        assert daemon_client.discover(project) is None
        assert not path.exists(), "stale discovery file should be removed"

    def test_version_mismatch_falls_back(
        self, project: pathlib.Path, server: daemon_mod.DaemonServer
    ):
        path = server.file_path
        raw = json.loads(path.read_text())
        raw["version"] = "0.0.0-other"
        path.write_text(json.dumps(raw))
        assert daemon_client.discover(project) is None
        assert path.exists(), "live daemon's file must not be removed"

    def test_chrome_port_env_disables_daemon(
        self,
        project: pathlib.Path,
        server: daemon_mod.DaemonServer,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv(daemon_client.CHROME_PORT_ENV, "9222")
        assert daemon_client.discover(project) is None

    def test_validate_info_health_check(self, server: daemon_mod.DaemonServer):
        info = _info(server)
        assert daemon_client.validate_info(info)


class TestAuth:
    def test_bad_token_rejected(self, server: daemon_mod.DaemonServer):
        info = _info(server)
        forged = daemon_client.DaemonInfo(
            path=info.path,
            port=info.port,
            pid=info.pid,
            version=info.version,
            token="wrong-token",
            started=info.started,
        )
        with pytest.raises(daemon_client.DaemonUnavailable):
            daemon_client.request(forged, "GET", "/status")

    def test_good_token_accepted(self, server: daemon_mod.DaemonServer):
        status = _status(server)
        assert status["ok"] is True
        assert status["pool"]["max"] == 2


class TestLifecycle:
    def test_start_writes_discovery_file(
        self, project: pathlib.Path, server: daemon_mod.DaemonServer
    ):
        info = _info(server)
        assert info.pid == os.getpid()
        assert info.version == daemon_client.colight_version()
        assert (project / ".colight_cache" / ".gitignore").read_text() == "*\n"

    def test_shutdown_removes_file_and_is_idempotent(self, project: pathlib.Path):
        daemon = daemon_mod.DaemonServer(project, idle_timeout=300.0)
        daemon.start()
        path = daemon.file_path
        assert path.exists()
        daemon.shutdown()
        daemon.shutdown()
        assert not path.exists()
        assert daemon.stopped

    def test_idle_self_shutdown(self, project: pathlib.Path):
        daemon = daemon_mod.DaemonServer(project, idle_timeout=0.3)
        daemon.start()
        try:
            deadline = time.time() + 5.0
            while time.time() < deadline and not daemon.stopped:
                time.sleep(0.05)
            assert daemon.stopped, "daemon should shut itself down when idle"
            assert not daemon.file_path.exists()
        finally:
            daemon.shutdown()

    def test_shutdown_endpoint(self, project: pathlib.Path):
        daemon = daemon_mod.DaemonServer(project, idle_timeout=300.0)
        daemon.start()
        try:
            daemon_client.request(_info(daemon), "POST", "/shutdown")
            deadline = time.time() + 5.0
            while time.time() < deadline and not daemon.stopped:
                time.sleep(0.05)
            assert daemon.stopped
        finally:
            daemon.shutdown()

    def test_cli_start_status_stop(self, project: pathlib.Path):
        def run_cli(*args: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                [sys.executable, "-m", "colight_cli", *args],
                cwd=project,
                capture_output=True,
                text=True,
                timeout=60,
            )

        started = run_cli("daemon", "start", "--idle-timeout", "120")
        assert started.returncode == 0, started.stderr
        assert "daemon started" in started.stdout
        try:
            status = run_cli("daemon", "status", "--json")
            assert status.returncode == 0, status.stderr
            payload = json.loads(status.stdout)
            assert payload["running"] is True
            assert payload["pool"]["instances"] == 0  # Chrome is lazy
        finally:
            stopped = run_cli("daemon", "stop")
        assert stopped.returncode == 0, stopped.stderr
        assert not daemon_client.daemon_file_path(project).exists()
        after = run_cli("daemon", "status")
        assert after.returncode == 1
        assert "no daemon running" in after.stdout


class TestSceneKey:
    def test_stable_and_content_sensitive(self, project: pathlib.Path):
        path = project / "scene.py"
        path.write_text(SCENE)
        key_a = daemon_client.scene_key(path, None, 400, 400, 1.0)
        assert key_a == daemon_client.scene_key(path, None, 400, 400, 1.0)
        assert key_a != daemon_client.scene_key(path, None, 800, 400, 1.0)
        path.write_text(SCENE.replace("[1, 0, 0]", "[0.5, 0, 0]"))
        assert key_a != daemon_client.scene_key(path, None, 400, 400, 1.0)

    def test_colight_target_hashes_bytes(self, project: pathlib.Path):
        artifact = project / "vis.colight"
        artifact.write_bytes(b"one")
        key_a = daemon_client.scene_key(artifact, None, 400, None, 1.0)
        artifact.write_bytes(b"two")
        assert key_a != daemon_client.scene_key(artifact, None, 400, None, 1.0)


@pytest.fixture
def scene_project(project: pathlib.Path) -> Tuple[pathlib.Path, pathlib.Path]:
    scene = project / "scene.py"
    scene.write_text(SCENE)
    return project, scene


class TestRoutedCommands:
    """End-to-end: CLI commands transparently served by a live daemon."""

    def test_screenshot_routed_and_byte_identical(
        self, scene_project: Tuple[pathlib.Path, pathlib.Path]
    ):
        _require_renderer()
        project, scene = scene_project
        runner = CliRunner()

        direct = runner.invoke(
            cli_main,
            [
                "screenshot",
                str(scene),
                "-o",
                str(project / "direct.png"),
                "--no-daemon",
                "--json",
                *SIZE_ARGS,
            ],
        )
        assert direct.exit_code == 0, direct.output

        daemon = daemon_mod.DaemonServer(project, idle_timeout=300.0, pool_size=2)
        daemon.start()
        try:
            routed = runner.invoke(
                cli_main,
                [
                    "screenshot",
                    str(scene),
                    "-o",
                    str(project / "routed.png"),
                    "--json",
                    *SIZE_ARGS,
                ],
            )
            assert routed.exit_code == 0, routed.output
            status = _status(daemon)
            # The request demonstrably hit the daemon...
            assert status["requests"].get("/screenshot") == 1
            assert status["pool"]["launches"] == 1
            # ...and produced the identical bytes direct mode produces.
            assert (project / "routed.png").read_bytes() == (
                project / "direct.png"
            ).read_bytes()
            direct_payload = json.loads(direct.output)
            routed_payload = json.loads(routed.output)
            assert routed_payload["sha256"] == direct_payload["sha256"]
        finally:
            daemon.shutdown()

    def test_warm_scene_reuse_and_check_determinism(
        self, scene_project: Tuple[pathlib.Path, pathlib.Path]
    ):
        _require_renderer()
        project, scene = scene_project
        runner = CliRunner()
        daemon = daemon_mod.DaemonServer(project, idle_timeout=300.0, pool_size=2)
        daemon.start()
        try:
            pick_args = ["pick-at", str(scene), "120,200", "--json", *SIZE_ARGS]
            first = runner.invoke(cli_main, pick_args)
            assert first.exit_code == 0, first.output
            after_first = _status(daemon)
            assert after_first["requests"].get("/pick-at") == 1
            launches = after_first["pool"]["launches"]
            hits = after_first["warm"]["hits"]

            second = runner.invoke(cli_main, pick_args)
            assert second.exit_code == 0, second.output
            assert second.output == first.output
            after_second = _status(daemon)
            # Warm reuse: no new Chrome, no re-load — the scene was served
            # from the LRU (and the .py was not re-evaluated: a warm hit
            # needs no visual payload at all).
            assert after_second["pool"]["launches"] == launches
            assert after_second["warm"]["hits"] == hits + 1
            assert after_second["warm"]["entries"] >= 1

            # Determinism THROUGH the daemon path (--check renders twice,
            # the recheck in a genuinely fresh tab).
            check = runner.invoke(
                cli_main,
                [
                    "screenshot",
                    str(scene),
                    "-o",
                    str(project / "check.png"),
                    "--check",
                    "--json",
                    *SIZE_ARGS,
                ],
            )
            assert check.exit_code == 0, check.output
            payload = json.loads(check.output)
            assert payload["deterministic"] is True
            assert _status(daemon)["requests"].get("/screenshot") == 1
        finally:
            daemon.shutdown()

    def test_pick_where_routed(self, scene_project: Tuple[pathlib.Path, pathlib.Path]):
        _require_renderer()
        project, scene = scene_project
        runner = CliRunner()
        daemon = daemon_mod.DaemonServer(project, idle_timeout=300.0, pool_size=2)
        daemon.start()
        try:
            result = runner.invoke(
                cli_main,
                [
                    "pick-where",
                    str(scene),
                    "--component",
                    "Ellipsoid",
                    "--out",
                    str(project / "overlay.png"),
                    "--json",
                    *SIZE_ARGS,
                ],
            )
            assert result.exit_code == 0, result.output
            payload = json.loads(result.output)
            assert payload["visible_pixels"] > 0
            assert (project / "overlay.png").exists()
            assert _status(daemon)["requests"].get("/pick-where") == 1
        finally:
            daemon.shutdown()

    def test_parallel_requests_use_pool_of_two(
        self, scene_project: Tuple[pathlib.Path, pathlib.Path]
    ):
        _require_renderer()
        project, scene = scene_project
        from colight.cli_tools import screenshot_tools

        data, buffers, _block = screenshot_tools.resolve_visual(scene, None)
        visual = daemon_client._encode_visual(data, buffers)

        daemon = daemon_mod.DaemonServer(project, idle_timeout=300.0, pool_size=2)
        daemon.start()
        info = _info(daemon)
        results: list = [None, None]

        def render(slot: int) -> None:
            results[slot] = daemon_client.request(
                info,
                "POST",
                "/render",
                {"visual": visual, "width": 400, "height": 400, "dpr": 1.0},
            )

        try:
            threads = [threading.Thread(target=render, args=(slot,)) for slot in (0, 1)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=120)
            assert all(r is not None and "png" in r for r in results)
            # Both renders returned identical pixels...
            assert results[0]["png"] == results[1]["png"]
            status = _status(daemon)
            # ...and the concurrent checkout forced a second isolated Chrome.
            assert status["pool"]["launches"] == 2
            assert status["pool"]["busy"] == 0
        finally:
            daemon.shutdown()

    def test_verify_pixels_via_daemon(
        self, scene_project: Tuple[pathlib.Path, pathlib.Path]
    ):
        _require_renderer()
        project, scene = scene_project
        runner = CliRunner()

        pinned = runner.invoke(
            cli_main, ["verify", str(scene), "--update", "--no-daemon", "--json"]
        )
        assert pinned.exit_code == 0, pinned.output

        daemon = daemon_mod.DaemonServer(project, idle_timeout=300.0, pool_size=2)
        daemon.start()
        try:
            verified = runner.invoke(cli_main, ["verify", str(scene), "--json"])
            assert verified.exit_code == 0, verified.output
            payload = json.loads(verified.output)
            assert payload["ok"] is True
            block = payload["targets"][0]["blocks"][0]
            assert block["status"] == "match"
            assert block["pixels"]["match"] is True
            # The screenshot layer demonstrably rendered on the daemon.
            assert _status(daemon)["requests"].get("/render") == 1
        finally:
            daemon.shutdown()
