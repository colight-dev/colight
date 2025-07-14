"""Tests for the cache manager."""

import time

from colight_live.cache_manager import CacheManager


class TestCacheManager:
    """Test cache management functionality."""

    def test_basic_operations(self):
        """Test basic add, access, remove operations."""
        cache = CacheManager(max_size_mb=10)

        # Add entry
        cache.add_entry("key1", "file1.py", 1024)
        assert "key1" in cache.entries
        assert "file1.py" in cache.file_entries
        assert cache.total_size == 1024

        # Access entry
        assert cache.access_entry("key1")
        assert cache.hit_count == 1
        assert cache.miss_count == 0

        # Access non-existent
        assert not cache.access_entry("key2")
        assert cache.miss_count == 1

        # Remove entry
        cache.remove_entry("key1")
        assert "key1" not in cache.entries
        assert cache.total_size == 0

    def test_file_tracking(self):
        """Test that entries are properly tracked by file."""
        cache = CacheManager()

        # Add entries from different files
        cache.add_entry("key1", "file1.py", 1024)
        cache.add_entry("key2", "file1.py", 1024)
        cache.add_entry("key3", "file2.py", 1024)

        assert len(cache.file_entries["file1.py"]) == 2
        assert len(cache.file_entries["file2.py"]) == 1

        # Clear file cache
        cache.clear_file_cache("file1.py")
        assert "file1.py" not in cache.file_entries
        assert "key1" not in cache.entries
        assert "key2" not in cache.entries
        assert "key3" in cache.entries

    def test_eviction_marking(self):
        """Test marking files for eviction."""
        cache = CacheManager()

        # Add entries
        cache.add_entry("key1", "file1.py", 1024)
        cache.add_entry("key2", "file2.py", 1024)

        # Mark file1 for eviction
        cache.mark_file_for_eviction("file1.py")
        assert "file1.py" in cache.marked_for_eviction

        # Evict marked files (non-hot entries)
        cache.evict_marked_files()
        assert "key1" not in cache.entries
        assert "key2" in cache.entries
        assert cache.eviction_count == 1

    def test_hot_entry_protection(self):
        """Test that hot entries are protected from eviction."""
        cache = CacheManager(hot_threshold_seconds=300)

        # Add entry and make it hot
        cache.add_entry("key1", "file1.py", 1024)
        for _ in range(10):  # Access multiple times to make it hot
            cache.access_entry("key1")

        assert cache.entries["key1"].is_hot

        # Mark for eviction
        cache.mark_file_for_eviction("file1.py")

        # Evict without force - hot entry should survive
        cache.evict_marked_files(force=False)
        assert "key1" in cache.entries

        # Evict with force - hot entry should be removed
        cache.evict_marked_files(force=True)
        assert "key1" not in cache.entries

    def test_size_limit_eviction(self):
        """Test automatic eviction when size limit exceeded."""
        cache = CacheManager(max_size_mb=1)  # 1MB limit

        # Add entries that exceed limit
        cache.add_entry("key1", "file1.py", 500 * 1024)  # 500KB
        cache.add_entry("key2", "file2.py", 400 * 1024)  # 400KB
        cache.add_entry("key3", "file3.py", 300 * 1024)  # 300KB

        # Should have evicted oldest entries to stay under 1MB
        assert cache.total_size <= 1024 * 1024
        assert cache.eviction_count > 0

    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = CacheManager(max_size_mb=1)

        # Add entries
        cache.add_entry("key1", "file1.py", 400 * 1024)  # 400KB
        time.sleep(0.01)
        cache.add_entry("key2", "file2.py", 400 * 1024)  # 400KB
        time.sleep(0.01)
        cache.add_entry("key3", "file3.py", 400 * 1024)  # 400KB

        # Access key1 to make it more recent than key2
        cache.access_entry("key1")

        # Add another entry to trigger eviction
        cache.add_entry("key4", "file4.py", 400 * 1024)

        # With 1MB limit and 4x400KB entries, only 2 can remain
        # Should keep the most recently used/added
        assert len(cache.entries) == 2
        assert cache.total_size <= 1024 * 1024

        # key1 was accessed (not just created), so it should remain
        assert "key1" in cache.entries
        # key4 was just added, so it should remain
        assert "key4" in cache.entries

    def test_statistics(self):
        """Test cache statistics."""
        cache = CacheManager()

        # Add entries
        cache.add_entry("key1", "file1.py", 1024)
        cache.add_entry("key2", "file2.py", 2048)

        # Access entries
        cache.access_entry("key1")
        cache.access_entry("key1")
        cache.access_entry("missing")

        # Mark file for eviction
        cache.mark_file_for_eviction("file2.py")

        stats = cache.get_stats()
        assert stats["total_entries"] == 2
        assert stats["watched_entries"] == 1  # only file1.py is watched
        assert stats["marked_files"] == 1
        assert stats["hit_count"] == 2
        assert stats["miss_count"] == 1
        assert stats["hit_rate"] == 2 / 3
        assert stats["size_mb"] == (1024 + 2048) / 1024 / 1024

    def test_entry_size_estimation(self):
        """Test entry size estimation."""
        cache = CacheManager()

        # Mock result object
        class MockResult:
            output = "Hello world"
            error = None
            colight_bytes = b"x" * 1000

        result = MockResult()
        size = cache.estimate_entry_size(result)

        # Should include base + output + colight_bytes + overhead
        assert size >= 1024 + len("Hello world") + 1000 + 512
