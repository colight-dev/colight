#!/usr/bin/env python3
"""
Test suite for Chrome DevTools functionality
"""

import time
import json
import urllib.request
from pathlib import Path

from colight.chrome_devtools import ChromeContext, shutdown_chrome


def get_open_tabs(port):
    """Get list of open tabs from Chrome DevTools"""
    try:
        response = urllib.request.urlopen(f"http://localhost:{port}/json", timeout=1)
        tabs = json.loads(response.read())
        # Filter out the default data:, tab that Chrome creates on startup
        return [
            tab
            for tab in tabs
            if tab.get("type") == "page" and tab.get("url") != "data:,"
        ]
    except Exception:
        return []


def verify_tab_cleanup(port, expected_count=0, debug=False):
    """Verify that the expected number of tabs are open"""
    tabs = get_open_tabs(port)
    actual_count = len(tabs)

    if debug:
        print(f"   Open tabs: {actual_count} (expected: {expected_count})")
        for tab in tabs:
            print(f"     - {tab.get('id', 'unknown')}: {tab.get('title', 'no title')}")

    return actual_count == expected_count


def test_chrome_devtools():
    """Comprehensive test of Chrome DevTools functionality

    Run this manually to test:
    - Chrome startup timing
    - Tab creation and cleanup
    - HTML loading and serving
    - Screenshot capture
    - WebGPU support detection
    - Multiple context handling
    - Tab cleanup verification
    """

    print("=" * 60)
    print("Chrome DevTools Test Suite")
    print("=" * 60)

    # Ensure clean state
    shutdown_chrome(debug=True)

    # Test 1: Basic Chrome startup and shutdown
    print("\n1. Testing basic Chrome startup...")
    start_time = time.time()

    with ChromeContext(width=800, height=600, debug=True, keep_alive=0) as chrome:
        startup_time = time.time() - start_time
        print(f"✅ Chrome started in {startup_time:.3f}s")

        # Verify initial tab count
        assert verify_tab_cleanup(
            chrome.port, expected_count=1, debug=True
        ), "Expected 1 tab after context creation"

        # Test 2: HTML loading and screenshot
        print("\n2. Testing HTML loading and screenshot...")
        test_html = """
        <html>
        <head><title>Chrome DevTools Test</title></head>
        <body style="background: linear-gradient(45deg, #ff6b6b, #4ecdc4); 
                     margin: 0; padding: 20px; font-family: Arial, sans-serif;">
            <h1 style="color: white; text-align: center;">Chrome DevTools Test</h1>
            <p style="color: white; text-align: center;">This is a test page for Chrome DevTools functionality</p>
            <div id="test-div" style="background: rgba(255,255,255,0.2); 
                                     padding: 20px; border-radius: 10px; 
                                     margin: 20px auto; max-width: 400px;">
                <p style="color: white; margin: 0;">Test content loaded successfully!</p>
            </div>
        </body>
        </html>
        """

        chrome.load_html(test_html)
        print("✅ HTML loaded successfully")

        # Capture screenshot
        image_data = chrome.capture_image()
        print(f"✅ Screenshot captured ({len(image_data)} bytes)")

        # Save screenshot
        test_dir = Path("./test-artifacts")
        test_dir.mkdir(exist_ok=True)
        screenshot_path = test_dir / "chrome_test_screenshot.png"
        with open(screenshot_path, "wb") as f:
            f.write(image_data)
        print(f"✅ Screenshot saved to {screenshot_path}")

    # Verify tab cleanup after context closes
    print("\n   Verifying tab cleanup after context close...")
    assert verify_tab_cleanup(
        chrome.port, expected_count=0, debug=True
    ), "Expected 0 tabs after context close"
    print("✅ Tab cleanup verified")

    print("✅ Chrome shutdown complete")

    # Test 3: WebGPU support detection
    print("\n3. Testing WebGPU support detection...")
    with ChromeContext(width=400, height=300, debug=True, keep_alive=0) as chrome:
        webgpu_info = chrome.check_webgpu_support()
        if webgpu_info.get("supported"):
            print(
                f"✅ WebGPU supported: {webgpu_info.get('adapter', {}).get('name', 'Unknown')}"
            )
            features = webgpu_info.get("features", [])
            if features:
                print(
                    f"   Features: {', '.join(features[:5])}{'...' if len(features) > 5 else ''}"
                )
        else:
            print(
                f"❌ WebGPU not supported: {webgpu_info.get('reason', 'Unknown reason')}"
            )

    # Verify tab cleanup
    assert verify_tab_cleanup(
        chrome.port, expected_count=0, debug=True
    ), "Expected 0 tabs after WebGPU test"

    # Test 4: Multiple contexts and tab cleanup
    print("\n4. Testing multiple contexts and tab cleanup...")
    contexts = []
    for i in range(3):
        print(f"   Creating context {i+1}/3...")
        ctx = ChromeContext(width=300, height=200, debug=False, keep_alive=0)
        ctx.start()
        ctx.load_html(
            f"<html><body style='background: #{0xff0000 + i*0x111111:06x};'>Context {i+1}</body></html>"
        )
        contexts.append(ctx)

    # Verify we have 3 tabs open
    assert verify_tab_cleanup(
        contexts[0].port, expected_count=3, debug=True
    ), "Expected 3 tabs after creating 3 contexts"

    print("   All contexts created, now closing them...")
    for i, ctx in enumerate(contexts):
        print(f"   Closing context {i+1}/3...")
        ctx.stop()

    # Verify all tabs are closed
    print("   Verifying all tabs are closed...")
    assert verify_tab_cleanup(
        contexts[0].port, expected_count=0, debug=True
    ), "Expected 0 tabs after closing all contexts"

    print("✅ Multiple contexts test complete")

    # Test 5: Explicit shutdown
    print("\n5. Testing explicit shutdown...")
    with ChromeContext(width=400, height=300, debug=True, keep_alive=5) as chrome:
        chrome.load_html("<html><body>Test for explicit shutdown</body></html>")
        print("   Context created, testing explicit shutdown...")

    # Wait a moment then explicitly shutdown
    time.sleep(0.5)
    shutdown_chrome(debug=True)
    print("✅ Explicit shutdown test complete")

    # Final verification
    print("\n6. Final cleanup verification...")
    try:
        # Try to connect to Chrome - should fail if properly shut down
        urllib.request.urlopen("http://localhost:9222/json", timeout=1)
        assert False, "Chrome is still running after shutdown"
    except Exception:
        print("✅ Chrome properly shut down")

    print("\n" + "=" * 60)
    print("All tests completed successfully! 🎉")
    print("=" * 60)
    print(f"Check {test_dir} for generated test artifacts")


def test_tab_cleanup_stress():
    """Stress test for tab cleanup - create many contexts rapidly"""
    print("\n" + "=" * 60)
    print("Tab Cleanup Stress Test")
    print("=" * 60)

    # Ensure clean state
    shutdown_chrome(debug=True)

    num_contexts = 10
    contexts = []

    print(f"Creating {num_contexts} contexts rapidly...")
    for i in range(num_contexts):
        ctx = ChromeContext(width=200, height=150, debug=False, keep_alive=0)
        ctx.start()
        ctx.load_html(f"<html><body>Stress Test Context {i+1}</body></html>")
        contexts.append(ctx)

    # Verify all tabs are open
    assert verify_tab_cleanup(
        contexts[0].port, expected_count=num_contexts, debug=True
    ), f"Expected {num_contexts} tabs"

    print(f"Closing {num_contexts} contexts...")
    for ctx in contexts:
        ctx.stop()

    # Verify all tabs are closed
    assert verify_tab_cleanup(
        contexts[0].port, expected_count=0, debug=True
    ), "Expected 0 tabs after stress test"

    print("✅ Stress test completed successfully")


if __name__ == "__main__":
    # Run the comprehensive test
    success = test_chrome_devtools()

    if success:
        # Run stress test
        test_tab_cleanup_stress()

    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed!")
    print("=" * 60)
