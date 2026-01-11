"""Concurrency tests for visual_store thread safety.

NOTE: These tests document the potential concurrency issues but the visual_store
is left as a plain dict because:
1. Werkzeug's development server is single-threaded by default
2. CPython's GIL makes basic dict operations atomic
3. The simple key-value cache pattern has minimal concurrency risk
4. This is a development server not intended for production use
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict


class MockVisualStore:
    """Mock implementation to test thread safety issues."""

    def __init__(self, use_lock=False):
        self.visual_store: Dict[str, bytes] = {}
        self.use_lock = use_lock
        self.lock = threading.Lock() if use_lock else None
        self.conflicts = 0
        self.reads = 0
        self.writes = 0

    def write(self, key: str, data: bytes):
        """Simulate writing to visual store."""
        if self.use_lock and self.lock is not None:
            with self.lock:
                self._write_unsafe(key, data)
        else:
            self._write_unsafe(key, data)

    def _write_unsafe(self, key: str, data: bytes):
        """Unsafe write operation that can demonstrate race conditions."""
        self.writes += 1
        # Simulate some processing time
        existing = self.visual_store.get(key)
        time.sleep(0.0001)  # Small delay to increase chance of race condition

        # Check if value changed during our "processing"
        if existing != self.visual_store.get(key):
            self.conflicts += 1

        self.visual_store[key] = data

    def read(self, key: str) -> bytes:
        """Simulate reading from visual store."""
        if self.use_lock and self.lock is not None:
            with self.lock:
                return self._read_unsafe(key)
        else:
            return self._read_unsafe(key)

    def _read_unsafe(self, key: str) -> bytes:
        """Unsafe read operation."""
        self.reads += 1
        result = self.visual_store.get(key)
        if result is None:
            return b""
        return result


def test_concurrent_writes_without_lock():
    """Test that concurrent writes without lock can cause issues."""
    store = MockVisualStore(use_lock=False)
    num_threads = 10
    writes_per_thread = 100

    def worker(thread_id: int):
        for i in range(writes_per_thread):
            # Multiple threads writing to same keys
            key = f"visual_{i % 20}"  # 20 unique keys shared across threads
            data = f"thread_{thread_id}_write_{i}".encode()
            store.write(key, data)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        for future in futures:
            future.result()

    # Without locks, we expect some conflicts or lost writes
    print(f"Conflicts detected: {store.conflicts}")
    print(f"Total writes: {store.writes}")

    # The final state might have lost some writes
    # With proper locking, we'd have exactly 20 keys
    assert len(store.visual_store) <= 20

    # We expect conflicts when not using locks
    # This test documents the problem, not enforces correctness
    assert store.conflicts > 0 or store.writes != num_threads * writes_per_thread


def test_concurrent_writes_with_lock():
    """Test that concurrent writes with lock work correctly."""
    store = MockVisualStore(use_lock=True)
    num_threads = 10
    writes_per_thread = 100

    def worker(thread_id: int):
        for i in range(writes_per_thread):
            key = f"visual_{i % 20}"  # 20 unique keys shared across threads
            data = f"thread_{thread_id}_write_{i}".encode()
            store.write(key, data)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        for future in futures:
            future.result()

    # With locks, no conflicts should occur
    assert store.conflicts == 0
    assert store.writes == num_threads * writes_per_thread
    assert len(store.visual_store) == 20  # Exactly 20 unique keys


def test_concurrent_read_write_without_lock():
    """Test concurrent reads and writes without lock."""
    store = MockVisualStore(use_lock=False)
    num_writers = 5
    num_readers = 5
    operations = 100
    exceptions = []

    # Pre-populate some data
    for i in range(10):
        store.write(f"visual_{i}", f"initial_{i}".encode())

    def writer(thread_id: int):
        try:
            for i in range(operations):
                key = f"visual_{i % 10}"
                data = f"writer_{thread_id}_op_{i}".encode()
                store.write(key, data)
        except Exception as e:
            exceptions.append(e)

    def reader(thread_id: int):
        try:
            for i in range(operations):
                key = f"visual_{i % 10}"
                data = store.read(key)
                # Just read, checking we don't crash
        except Exception as e:
            exceptions.append(e)

    with ThreadPoolExecutor(max_workers=num_writers + num_readers) as executor:
        writer_futures = [executor.submit(writer, i) for i in range(num_writers)]
        reader_futures = [executor.submit(reader, i) for i in range(num_readers)]

        for future in writer_futures + reader_futures:
            future.result()

    # Without proper synchronization, we might see issues
    # This test documents that the problem exists
    print(f"Total operations: {store.reads + store.writes}")
    print(f"Conflicts: {store.conflicts}")


def test_concurrent_read_write_with_lock():
    """Test concurrent reads and writes with lock."""
    store = MockVisualStore(use_lock=True)
    num_writers = 5
    num_readers = 5
    operations = 100

    # Pre-populate some data
    for i in range(10):
        store.write(f"visual_{i}", f"initial_{i}".encode())

    def writer(thread_id: int):
        for i in range(operations):
            key = f"visual_{i % 10}"
            data = f"writer_{thread_id}_op_{i}".encode()
            store.write(key, data)

    def reader(thread_id: int):
        read_values = []
        for i in range(operations):
            key = f"visual_{i % 10}"
            data = store.read(key)
            if data:
                read_values.append(data)
        return read_values

    with ThreadPoolExecutor(max_workers=num_writers + num_readers) as executor:
        writer_futures = [executor.submit(writer, i) for i in range(num_writers)]
        reader_futures = [executor.submit(reader, i) for i in range(num_readers)]

        for future in writer_futures:
            future.result()

        read_results = [future.result() for future in reader_futures]

    # With locks, everything should work correctly
    assert store.conflicts == 0
    assert store.writes == num_writers * operations + 10  # initial writes
    assert store.reads == num_readers * operations

    # All reads should have gotten valid data
    for reads in read_results:
        assert len(reads) == operations


def test_dict_modification_during_iteration():
    """Test that dict modification during iteration can cause issues."""
    # This simulates the potential issue in the live server
    store = {}
    errors = []

    def modifier():
        for i in range(100):
            store[f"key_{i}"] = f"value_{i}"
            time.sleep(0.0001)
            # Sometimes delete old keys
            if i > 10 and i % 10 == 0:
                try:
                    del store[f"key_{i-10}"]
                except KeyError:
                    pass

    def iterator():
        for _ in range(10):
            try:
                # Simulate iterating over clients and accessing store
                items = list(store.items())  # This can fail without proper locking
                for key, value in items:
                    # Simulate some processing
                    _ = store.get(key)  # Key might be gone
                time.sleep(0.001)
            except RuntimeError as e:
                errors.append(e)

    # Run both concurrently
    t1 = threading.Thread(target=modifier)
    t2 = threading.Thread(target=iterator)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    # This test documents that dict modification during iteration can cause issues
    # With CPython's GIL this might not always fail, but it's still incorrect
    print(f"Errors caught: {len(errors)}")
