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


def on_pre_build(config):
    if build_tailwind():
        # Ensure the tailwind output is included in extra_css
        if "extra_css" not in config:
            config["extra_css"] = []
        config["extra_css"].append("stylesheets/tailwind.css")
