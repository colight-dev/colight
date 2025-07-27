import logging
import subprocess
from pathlib import Path

log = logging.getLogger("mkdocs")

TAILWIND_INPUT = """
@tailwind base;
@tailwind components;
@tailwind utilities;
""" + Path("packages/colight/src/widget.css").read_text()


def build_tailwind():
    output_path = "docs/src/colight_docs/overrides/stylesheets/tailwind.css"

    try:
        subprocess.run(
            [
                "npx",
                "tailwindcss",
                "-i",
                "-",
                "-o",
                output_path,
                "--minify",
                "-c",
                "docs/src/colight_docs/overrides/tailwind.config.js",
            ],
            input=TAILWIND_INPUT.encode(),
            check=True,
        )
        log.info(f"Compiled Tailwind CSS to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to build Tailwind CSS: {e}")
        return False


def on_files(files, config):
    from mkdocs.structure.files import File

    # Copy js-dist files to docs overrides
    src_js_dist = Path("packages/colight/src/colight/js-dist")
    dest_js_base = "js/dist"  # Relative to overrides directory

    if src_js_dist.exists():
        # Iterate through all files in js-dist
        for src_file in src_js_dist.rglob("*"):
            if src_file.is_file():
                # Calculate relative path within js-dist
                rel_path = src_file.relative_to(src_js_dist)

                # Create destination path relative to docs directory
                dest_uri = f"{dest_js_base}/{rel_path}"

                # Create a File object for this static file
                file_obj = File.generated(
                    config, src_uri=dest_uri, abs_src_path=str(src_file.absolute())
                )

                # Add to the files collection
                files.append(file_obj)

        log.info(f"Added {src_js_dist} files to MkDocs files collection")
    else:
        log.warning(f"Source directory {src_js_dist} does not exist")

    return files


def on_pre_build(config):
    if build_tailwind():
        # Ensure the tailwind output is included in extra_css
        if "extra_css" not in config:
            config["extra_css"] = []
        config["extra_css"].append("stylesheets/tailwind.css")
