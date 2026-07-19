"""Shared Chrome instance lifecycle: a dead instance is closed on replace.

Exercises ``_acquire_shared_instance`` directly with stub processes — no
real Chrome needed.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import colight.chrome_devtools as chrome_devtools


def _dead_instance(prefix: str) -> chrome_devtools.ChromeInstance:
    """A ChromeInstance whose process has already exited."""
    user_data_dir = Path(tempfile.mkdtemp(prefix=prefix))
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return chrome_devtools.ChromeInstance(process, port=0, user_data_dir=user_data_dir)


def test_dead_shared_instance_is_closed_before_replacement(
    monkeypatch: pytest.MonkeyPatch,
):
    dead = _dead_instance("colight-chrome-dead-")
    replacement = _dead_instance("colight-chrome-new-")
    with chrome_devtools._lock:
        chrome_devtools._live_instances.add(dead)
    monkeypatch.setattr(chrome_devtools, "_shared_instance", dead)
    monkeypatch.setattr(
        chrome_devtools.ChromeInstance,
        "launch",
        classmethod(lambda cls, **kwargs: replacement),
    )

    try:
        with chrome_devtools._lock:
            acquired = chrome_devtools._acquire_shared_instance(400, None, False)

        assert acquired is replacement
        assert chrome_devtools._shared_instance is replacement
        # The dead instance was properly closed: profile dir removed and
        # instance dropped from the live registry.
        assert not dead.user_data_dir.exists()
        with chrome_devtools._lock:
            assert dead not in chrome_devtools._live_instances
    finally:
        replacement.close()
        with chrome_devtools._lock:
            chrome_devtools._live_instances.discard(dead)


def test_live_shared_instance_is_reused(monkeypatch: pytest.MonkeyPatch):
    alive_dir = Path(tempfile.mkdtemp(prefix="colight-chrome-alive-"))
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    alive = chrome_devtools.ChromeInstance(process, port=0, user_data_dir=alive_dir)
    monkeypatch.setattr(chrome_devtools, "_shared_instance", alive)
    monkeypatch.setattr(
        chrome_devtools.ChromeInstance,
        "launch",
        classmethod(lambda cls, **kwargs: pytest.fail("must not relaunch while alive")),
    )
    try:
        with chrome_devtools._lock:
            acquired = chrome_devtools._acquire_shared_instance(400, None, False)
        assert acquired is alive
        assert alive.user_data_dir.exists()
    finally:
        alive.close()
