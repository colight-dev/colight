"""Main builder module that coordinates parsing, execution, and generation."""

import pathlib
import subprocess
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Set, Union, get_args

from colight_prose.constants import DEFAULT_INLINE_THRESHOLD
from colight_prose.executor import DocumentExecutor
from colight_prose.file_resolver import find_files
from colight_prose.model import TagSet
from colight_prose.parser import parse_colight_file
from colight_prose.pep723 import detect_pep723_metadata, parse_dependencies
from colight_prose.pragma import parse_pragma_arg
from colight_prose.utils import merge_ignore_patterns

from .generator import HTMLGenerator, MarkdownGenerator, write_colight_files

# Define valid format types
FormatType = Literal["markdown", "html"]
VALID_FORMATS: Set[str] = set(get_args(FormatType))


def parse_formats_arg(formats: Union[str, Set[str], None]) -> Set[FormatType]:
    """Parse formats from comma-separated string.

    Args:
        formats: Either a string of comma-separated formats, a set of formats, or None

    Returns:
        Set of format strings (validation happens in BuildConfig)
    """
    if not formats:
        return {"markdown"}  # Default format

    # Handle both string and set inputs
    format_set = (
        formats
        if isinstance(formats, set)
        else {fmt.strip() for fmt in formats.split(",") if fmt.strip()}
    )

    return format_set or {"markdown"}  # type: ignore[return-value]


@dataclass
class BuildConfig:
    """Configuration for building a colight file."""

    verbose: bool = False
    pragma: set[str] = field(default_factory=set)
    formats: Set[FormatType] = field(default_factory=lambda: {"markdown"})
    continue_on_error: bool = True
    colight_output_path: Optional[str] = None
    colight_embed_path: Optional[str] = None
    inline_threshold: int = DEFAULT_INLINE_THRESHOLD
    in_subprocess: bool = False

    def __post_init__(self):
        """Validate formats after initialization."""
        if self.formats:
            # Convert to set of strings for comparison
            format_strings = {str(fmt) for fmt in self.formats}
            invalid_formats = format_strings - VALID_FORMATS
            if invalid_formats:
                raise ValueError(
                    f"Invalid formats: {', '.join(sorted(invalid_formats))}. "
                    f"Valid formats are: {', '.join(sorted(VALID_FORMATS))}"
                )

    @property
    def format(self) -> FormatType:
        """Get the primary format (for backward compatibility)."""
        return next(iter(self.formats))

    @property
    def hide_statements(self) -> bool:
        """Check if statements should be hidden."""
        return "hide-statements" in self.pragma

    @property
    def hide_visuals(self) -> bool:
        """Check if visuals should be hidden."""
        return "hide-visuals" in self.pragma

    @property
    def hide_code(self) -> bool:
        """Check if code should be hidden."""
        return "hide-code" in self.pragma

    def to_cli_args(self) -> List[str]:
        """Convert config to CLI arguments."""
        args = []

        if self.verbose:
            args.extend(["--verbose", "true"])

        # Convert pragma to comma-separated list
        if self.pragma:
            args.extend(["--pragma", ",".join(sorted(self.pragma))])

        # Convert formats to comma-separated list
        if self.formats and self.formats != {"markdown"}:
            args.extend(["--formats", ",".join(sorted(self.formats))])

        if not self.continue_on_error:
            args.extend(["--continue-on-error", "false"])

        if self.colight_output_path:
            args.extend(["--colight-output-path", self.colight_output_path])

        if self.colight_embed_path:
            args.extend(["--colight-embed-path", self.colight_embed_path])

        if self.inline_threshold != DEFAULT_INLINE_THRESHOLD:
            args.extend(["--inline-threshold", str(self.inline_threshold)])

        if self.in_subprocess:
            args.append("--in-subprocess")

        return args

    @classmethod
    def from_config_and_kwargs(
        cls, config: Optional["BuildConfig"] = None, **kwargs
    ) -> "BuildConfig":
        """Create a BuildConfig from an optional existing config and kwargs."""
        if config is None:
            # Parse pragma and formats if provided
            if "pragma" in kwargs:
                kwargs["pragma"] = parse_pragma_arg(kwargs["pragma"])
            if "formats" in kwargs:
                kwargs["formats"] = parse_formats_arg(kwargs["formats"])
            return cls(**kwargs)

        # Merge config with kwargs
        config_dict = {
            "verbose": config.verbose,
            "pragma": config.pragma,
            "formats": config.formats.copy(),
            "continue_on_error": config.continue_on_error,
            "colight_output_path": config.colight_output_path,
            "colight_embed_path": config.colight_embed_path,
            "inline_threshold": config.inline_threshold,
            "in_subprocess": config.in_subprocess,
        }
        config_dict.update(kwargs)

        # Parse pragma and formats if provided in kwargs
        if "pragma" in kwargs:
            config_dict["pragma"] = parse_pragma_arg(config_dict["pragma"])
        if "formats" in kwargs:
            config_dict["formats"] = parse_formats_arg(config_dict["formats"])

        return cls(**config_dict)


def get_output_path(input_path: pathlib.Path, format: FormatType) -> pathlib.Path:
    """Convert Python file path to output path with correct extension."""
    if input_path.suffix == ".py":
        # For regular .py files, replace .py with the output extension
        suffix = ".html" if format == "html" else ".md"
        return input_path.with_suffix(suffix)
    else:
        # Fallback for other files
        suffix = ".html" if format == "html" else ".md"
        return input_path.with_suffix(suffix)


def build_file(
    input_path: pathlib.Path,
    output_file: Optional[pathlib.Path] = None,
    output_dir: Optional[pathlib.Path] = None,
    config: Optional[BuildConfig] = None,
    **kwargs,
):
    """Build a single Python file.

    Args:
        input_path: Path to the Python file to build
        output_dir: Directory where output files will be written (for multiple formats)
        output_file: Specific output file path (for single format)
        config: BuildConfig object with build settings
        **kwargs: Additional keyword arguments to override config values

    Raises:
        ValueError: If both or neither output_dir and output_file are specified
    """
    # Must specify exactly one output location
    if (output_dir is None) == (output_file is None):
        raise ValueError("Must specify either output_dir or output_file, not both")
    # Create config from provided config or kwargs
    config = BuildConfig.from_config_and_kwargs(config, **kwargs)
    if not input_path.suffix == ".py":
        raise ValueError(f"Not a Python file: {input_path}")

    # Check if this is a PEP 723 file BEFORE doing any processing
    file_content = input_path.read_text(encoding="utf-8")
    pep723_metadata = detect_pep723_metadata(file_content)

    # Skip PEP 723 handling if we're already in a subprocess
    if pep723_metadata and not config.in_subprocess:
        if config.verbose:
            print("  Detected PEP 723 metadata - re-running with dependencies")

        # Parse the dependencies from PEP 723 metadata
        dependencies = parse_dependencies(pep723_metadata)

        # Build the uv run command
        # Get the colight-prose package root (3 levels up from builder.py)
        colight_prose_root = pathlib.Path(__file__).parent.parent.parent
        cmd = [
            "uv",
            "run",
            "--with-editable",
            str(colight_prose_root),  # colight-prose package root
        ]

        # Add each PEP 723 dependency
        for dep in dependencies:
            cmd.extend(["--with", dep])

        # Add the script flag and colight-prose command
        cmd.extend(
            [
                "--",
                "python",
                "-m",
                "colight_cli",
                "build",
                str(input_path),
            ]
        )

        # Add output argument
        if output_file:
            cmd.extend(["-o", str(output_file)])
        else:
            assert output_dir is not None  # Guaranteed by the check at function start
            cmd.extend(["-o", str(output_dir)])

        # Add all CLI arguments from config
        config.in_subprocess = True  # Mark that subprocess will be in subprocess mode
        cmd.extend(config.to_cli_args())

        if config.verbose:
            print(f"  Running: {' '.join(cmd)}")

        # Run the command and let output pass through in real-time
        result = subprocess.run(cmd)

        # Return instead of sys.exit to allow tests to continue
        # The subprocess has completed, so we just return to prevent further processing
        return

    # Not a PEP 723 file or already in PEP 723 environment - continue with normal execution
    if config.verbose:
        if output_file:
            print(f"Building {input_path} -> {output_file}")
        else:
            print(f"Building {input_path} -> {output_dir}/")
    try:
        # Parse the file
        document = parse_colight_file(input_path)
        if config.verbose:
            print(f"Found {len(document.blocks)} blocks")
            if document.tags.flags:
                print(f"  File tags: {document.tags.flags}")
    except Exception as e:
        if config.verbose:
            print(f"Parse error: {e}")
        # Create a minimal output file with error message
        if output_file:
            # Single file output
            output_file.parent.mkdir(parents=True, exist_ok=True)
            error_content = f"# Parse Error\n\nCould not parse {input_path.name}: {e}\n"
            output_file.write_text(error_content)
        else:
            # Directory output - create error file for first format
            assert output_dir is not None  # Guaranteed by the check at function start
            output_dir.mkdir(parents=True, exist_ok=True)
            error_format = next(iter(config.formats))
            error_file = output_dir / get_output_path(
                pathlib.Path(input_path.name), error_format
            )
            error_content = f"# Parse Error\n\nCould not parse {input_path.name}: {e}\n"
            error_file.write_text(error_content)
        return

    # Apply pragma from config
    if config.pragma:
        document.tags = document.tags | TagSet(frozenset(config.pragma))

    # Setup execution environment
    # Default templates if not provided
    embed_template = (
        config.colight_embed_path or "{basename}_colight/block-{block:03d}.colight"
    )

    # Execute the document
    executor = DocumentExecutor(verbose=config.verbose)
    results, _ = executor.execute(document, str(input_path))

    # Determine colight directory and basename
    if output_file:
        colight_dir = output_file.parent / f"{output_file.stem}_colight"
        colight_basename = output_file.stem
    else:
        assert output_dir is not None  # Guaranteed by the check at function start
        colight_dir = output_dir / f"{input_path.stem}_colight"
        colight_basename = input_path.stem

    if config.verbose:
        print(f"  Writing .colight files to: {colight_dir}")

    # Prepare path context for templates
    path_context = {
        "basename": colight_basename,
        "filename": input_path.name,
    }

    # Write colight files if needed
    colight_paths = write_colight_files(colight_dir, results, basename=colight_basename)

    # Check if we had any errors
    execution_errors = [r.error for r in results if r.error]
    if execution_errors and not config.continue_on_error:
        if config.verbose:
            print("  Execution stopped due to errors")

    # For visualizations that should be inlined, update the results
    inline_visuals = []
    for i, result in enumerate(results):
        if result.colight_bytes and len(result.colight_bytes) < config.inline_threshold:
            inline_visuals.append(result.colight_bytes)
        else:
            inline_visuals.append(colight_paths[i] if i < len(colight_paths) else None)

    # Generate output
    if output_file:
        # Single file output
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Determine format from file extension
        formats_to_build: List[FormatType]
        if output_file.suffix == ".html":
            formats_to_build = ["html"]
        elif output_file.suffix == ".md":
            formats_to_build = ["markdown"]
        else:
            # No/unknown extension - throw if more than one format specified
            if len(config.formats) > 1:
                raise ValueError(
                    f"Output file {output_file} has no/unknown extension and multiple formats specified: "
                    f"{', '.join(sorted(config.formats))}. Please specify a single format or use a file with .html/.md extension."
                )
            formats_to_build = [next(iter(config.formats))]
    else:
        # Directory output - build all requested formats
        assert output_dir is not None  # Guaranteed by the check at function start
        output_dir.mkdir(parents=True, exist_ok=True)
        formats_to_build = list(config.formats)

    # Build each format
    for fmt in formats_to_build:
        if output_file:
            final_output_path = output_file
        else:
            assert output_dir is not None  # Guaranteed by the check at function start
            final_output_path = output_dir / get_output_path(
                pathlib.Path(input_path.name), fmt
            )

        if fmt == "html":
            title = input_path.stem.replace(".colight", "").replace("_", " ").title()
            html_generator = HTMLGenerator(
                colight_dir,
                embed_path_template=embed_template,
                inline_threshold=config.inline_threshold,
            )
            html_content = html_generator.generate(
                document, results, title, path_context
            )
            final_output_path.write_text(html_content)
        else:
            markdown_generator = MarkdownGenerator(
                colight_dir,
                embed_path_template=embed_template,
                inline_threshold=config.inline_threshold,
            )
            markdown_content = markdown_generator.generate(
                document, results, path_context
            )
            final_output_path.write_text(markdown_content)

        if config.verbose:
            print(f"Generated {final_output_path}")


def build_directory(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    config: Optional[BuildConfig] = None,
    include: Optional[List[str]] = None,
    ignore: Optional[List[str]] = None,
    **kwargs,
):
    """Build all Python files in a directory matching the patterns.

    Args:
        input_dir: Directory containing Python files to build
        output_dir: Directory where output files will be written
        config: BuildConfig object with build settings
        include: List of glob patterns to include
        ignore: List of glob patterns to ignore
        **kwargs: Additional keyword arguments to override config values
    """
    # Create config from provided config or kwargs
    config = BuildConfig.from_config_and_kwargs(config, **kwargs)

    if config.verbose:
        print(f"Building directory {input_dir} -> {output_dir}")

    # Default to .py files if no include patterns specified
    if include is None:
        include = ["*.py"]

    # Find all matching files
    python_files = find_files(input_dir, include, merge_ignore_patterns(ignore))

    # Remove duplicates and sort
    python_files = sorted(set(python_files))

    if config.verbose:
        print(f"Found {len(python_files)} Python files")

    # Build each file
    for python_file in python_files:
        try:
            # Calculate relative output path
            rel_path = python_file.relative_to(input_dir)

            # Build to the appropriate subdirectory
            output_subdir = output_dir / rel_path.parent

            build_file(
                python_file,
                output_dir=output_subdir,
                config=config,
            )
        except Exception as e:
            print(f"Error building {python_file}: {e}")
            if config.verbose:
                import traceback

                traceback.print_exc()
