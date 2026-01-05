"""Build utilities for colight-prose."""

import json
import pathlib
from typing import Dict, Optional, Union

import colight_prose.static.builder as builder

from .builder import BuildConfig


class BuildHelper:
    """Helper class for build-related operations."""

    @staticmethod
    def get_metadata_path(html_path: pathlib.Path) -> pathlib.Path:
        """Get the metadata file path for an HTML file.

        Args:
            html_path: Path to the HTML file

        Returns:
            Path to the metadata file
        """
        return html_path.with_suffix(".meta.json")

    @staticmethod
    def save_build_metadata(
        html_path: pathlib.Path, source_path: pathlib.Path, config: BuildConfig
    ) -> None:
        """Save build metadata for caching.

        Args:
            html_path: Path to the built HTML file
            source_path: Path to the source file
            config: Build configuration
        """
        metadata = {
            "source_mtime": source_path.stat().st_mtime,
            "pragma": sorted(list(config.pragma)),
            "source_path": str(source_path),
        }

        meta_path = BuildHelper.get_metadata_path(html_path)
        meta_path.write_text(json.dumps(metadata, indent=2))

    @staticmethod
    def load_build_metadata(html_path: pathlib.Path) -> Optional[Dict]:
        """Load build metadata if it exists.

        Args:
            html_path: Path to the HTML file

        Returns:
            Metadata dictionary if exists and valid, None otherwise
        """
        meta_path = BuildHelper.get_metadata_path(html_path)
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                return None
        return None

    @staticmethod
    def should_rebuild_with_metadata(
        source_path: pathlib.Path, html_path: pathlib.Path, config: BuildConfig
    ) -> bool:
        """Check if rebuild is needed considering metadata.

        Args:
            source_path: Path to the source file
            html_path: Path to the HTML file
            config: Build configuration

        Returns:
            True if rebuild is needed, False otherwise
        """
        if not html_path.exists():
            return True

        metadata = BuildHelper.load_build_metadata(html_path)
        if not metadata:
            # No metadata, fall back to mtime check
            return source_path.stat().st_mtime > html_path.stat().st_mtime

        # Check if source file changed
        current_mtime = source_path.stat().st_mtime
        if current_mtime > metadata.get("source_mtime", 0):
            return True

        # Check if pragma tags changed
        current_pragmas = sorted(list(config.pragma))
        cached_pragmas = metadata.get("pragma", [])
        if current_pragmas != cached_pragmas:
            return True

        return False

    @staticmethod
    def build_file_if_stale(
        source_file: pathlib.Path,
        output_file: pathlib.Path,
        config: BuildConfig,
        error_format: str = "html",
    ) -> Optional[Union[str, Dict]]:
        """Build file if stale, returning content or error.

        Args:
            source_file: Path to the source file
            output_file: Path to the output file
            config: Build configuration
            error_format: Format for error responses ("html" or "json")

        Returns:
            None if build not needed or successful,
            error content (HTML string or dict) if build failed
        """
        if not BuildHelper.should_rebuild_with_metadata(
            source_file, output_file, config
        ):
            return None

        try:
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Build the file
            builder.build_file(source_file, output_file, config=config)

            # Save metadata
            BuildHelper.save_build_metadata(output_file, source_file, config)

            if config.verbose:
                print(f"Built {source_file} -> {output_file}")

            return None

        except Exception as e:
            error_msg = f"Build Error: {str(e)}"

            if error_format == "json":
                return {"error": error_msg, "type": "build_error"}
            else:
                return f"""<!DOCTYPE html>
<html>
<head>
    <title>Build Error</title>
    <style>
        body {{ font-family: monospace; margin: 40px; }}
        .error {{ color: red; white-space: pre-wrap; }}
    </style>
</head>
<body>
    <h1>Build Error</h1>
    <p>Failed to build: {source_file}</p>
    <pre class="error">{str(e)}</pre>
</body>
</html>"""
