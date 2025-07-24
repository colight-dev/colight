#!/usr/bin/env python3
"""
Integration test that runs the actual colight-prose server
and tests the JavaScript navigation against it.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def create_test_directory():
    """Create a test directory structure with Python files."""
    test_dir = tempfile.mkdtemp(prefix="colight_test_")

    # Create directory structure
    Path(test_dir, "src").mkdir()
    Path(test_dir, "src/components").mkdir()
    Path(test_dir, "tests").mkdir()

    # Create test files with content
    files = {
        "README.py": """# Test Project
            
This is a test project for integration testing.
""",
        "main.py": """# Main module

def main():
    print("Hello from main")
    
if __name__ == "__main__":
    main()
""",
        "src/utils.py": """# Utilities

def helper():
    return 42
""",
        "src/components/button.py": """# Button component

class Button:
    def __init__(self, label):
        self.label = label
""",
        "src/components/card.py": """# Card component

class Card:
    pass
""",
        "tests/test_main.py": """# Tests

def test_main():
    assert True
""",
    }

    for filepath, content in files.items():
        full_path = Path(test_dir, filepath)
        full_path.write_text(content)

    return test_dir


def start_server(test_dir, port=5555):
    """Start the colight-prose server watching the test directory."""
    # Start server process
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "colight_prose.cli",
            "watch",
            test_dir,
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for server to start
    for _ in range(30):  # 30 second timeout
        try:
            response = requests.get(f"http://localhost:{port}")
            if response.status_code == 200:
                return process
        except requests.exceptions.ConnectionError:
            time.sleep(1)

    # If we get here, server didn't start
    process.terminate()
    raise RuntimeError("Server failed to start in 30 seconds")


def test_navigation(driver, base_url):
    """Test various navigation scenarios."""
    tests_passed = 0
    tests_failed = 0

    def assert_element_text(selector, expected_text):
        nonlocal tests_passed, tests_failed
        try:
            element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
            actual_text = element.text
            if expected_text in actual_text:
                print(f"✓ Found '{expected_text}' in {selector}")
                tests_passed += 1
            else:
                print(
                    f"✗ Expected '{expected_text}' but got '{actual_text}' in {selector}"
                )
                tests_failed += 1
        except Exception as e:
            print(f"✗ Failed to find {selector}: {e}")
            tests_failed += 1

    def click_element(selector):
        element = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
        )
        element.click()
        time.sleep(0.5)  # Give time for navigation

    # Test 1: Root directory shows directory browser
    print("\n=== Test 1: Root directory browser ===")
    driver.get(base_url)
    time.sleep(2)  # Wait for initial load

    # Should see root directory contents
    assert_element_text("body", "src")
    assert_element_text("body", "tests")
    assert_element_text("body", "README.py")
    assert_element_text("body", "main.py")

    # Test 2: Navigate to a file
    print("\n=== Test 2: Navigate to file ===")
    click_element("div:contains('main.py')")
    time.sleep(2)

    # Should see file content
    assert_element_text("body", "Hello from main")
    assert_element_text(".topbar", "main.py")  # Breadcrumb should show file

    # Test 3: Navigate via breadcrumb
    print("\n=== Test 3: Breadcrumb navigation ===")
    click_element("button:contains('root')")
    time.sleep(2)

    # Should be back at root directory
    assert_element_text("body", "src")
    assert_element_text("body", "tests")

    # Test 4: Navigate to subdirectory
    print("\n=== Test 4: Subdirectory navigation ===")
    click_element("div:contains('src')")
    time.sleep(1)
    click_element("div:contains('components')")
    time.sleep(2)

    # Should see components directory
    assert_element_text("body", "button.py")
    assert_element_text("body", "card.py")

    # Test 5: Command bar navigation
    print("\n=== Test 5: Command bar ===")
    # Press Cmd+K (or Ctrl+K)
    body = driver.find_element(By.TAG_NAME, "body")
    if sys.platform == "darwin":
        body.send_keys(Keys.COMMAND + "k")
    else:
        body.send_keys(Keys.CONTROL + "k")
    time.sleep(1)

    # Search for utils
    search_input = driver.find_element(By.CSS_SELECTOR, "input[placeholder*='Search']")
    search_input.send_keys("utils")
    time.sleep(1)

    # Click on utils.py result
    click_element("div:contains('utils.py')")
    time.sleep(2)

    # Should navigate to utils.py
    assert_element_text("body", "def helper():")
    assert_element_text("body", "return 42")

    return tests_passed, tests_failed


def main():
    test_dir = None
    server_process = None
    driver = None

    try:
        # Create test directory
        print("Creating test directory...")
        test_dir = create_test_directory()
        print(f"Test directory: {test_dir}")

        # Start server
        print("\nStarting colight-prose server...")
        port = 5555
        server_process = start_server(test_dir, port)
        print(f"Server started on port {port}")

        # Setup Chrome driver
        print("\nSetting up browser...")
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(options=chrome_options)

        # Run tests
        print("\nRunning navigation tests...")
        base_url = f"http://localhost:{port}"
        tests_passed, tests_failed = test_navigation(driver, base_url)

        # Print results
        print(f"\n{'='*50}")
        print(f"Tests passed: {tests_passed}")
        print(f"Tests failed: {tests_failed}")
        print(f"{'='*50}")

        return 0 if tests_failed == 0 else 1

    except Exception as e:
        print(f"Error: {e}")
        return 1

    finally:
        # Cleanup
        if driver:
            driver.quit()

        if server_process:
            server_process.terminate()
            server_process.wait()

        if test_dir and os.path.exists(test_dir):
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    sys.exit(main())
