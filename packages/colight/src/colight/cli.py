"""Top-level CLI for Colight."""

import click

from colight_cli import publish as publish_command


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main():
    """Colight command-line interface."""


main.add_command(publish_command, "publish")


if __name__ == "__main__":
    main()
