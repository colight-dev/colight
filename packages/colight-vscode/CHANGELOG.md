# Change Log

All notable changes to the "colight.vscode" extension will be documented in this file.

Check [Keep a Changelog](http://keepachangelog.com/) for recommendations on how to structure this file.

## [Unreleased]

### Added

### Changed

### Fixed

## [0.1.26] - 2026-01-07

### Added

### Changed

### Fixed

## [0.1.24] - 2024-11-29

### Added

### Changed

### Fixed

## [0.1.23] - 2024-11-29

### Added

### Changed

### Fixed

## [0.1.22] - 2024-11-29

### Added

### Changed

### Fixed

## [0.1.21] - 2024-11-19

### Added

### Changed

### Fixed

## [0.1.20] - 2024-11-19

### Added

### Changed

### Fixed

## [0.1.19] - 2024-07-18

### Changed

- Removed `cell.type` from cell
- Renamed `renderMarkdownWithinCells` to `renderComments`
- `[markdown]` metadata is now completely ignored
- `evaluateAllCells` and `evaluateCellAndAbove` do not render comments, regardless of `renderComments` config

## [0.1.18] - 2024-07-16

### Changed

- Rename Grow Selection -> Expand Selection for consistency with conflicting commands

## [0.1.17] - 2024-07-16

### Fixed

- Fix 'when' predicate for grow/shrink

## [0.1.16] - 2024-07-15

### Added

- "Evaluate All Cells" command

### Fixed

- Import markdown when rendering a markdown cell

## [0.1.15] - 2024-07-15

### Fixed

- Fix package.json

## [0.1.14] - 2024-07-15

### Added

- Add "Grow Selection" and "Shrink Selection" commands

### Fixed

- More reliable markdown cell handling
- Fixed "Run Cell and Above" with explicit cells
- Rename "Generate Jupyter Notebook" to "Export Jupyter Notebook"

## [0.1.13] - 2024-07-12

### Changed

- Run Current Cell and Above only shows the result for the current cell.
- Comment paragraphs that begin with `# ---` or `# -*-` are rendered as code
- Explicit cell starting lines (`# %%`) are not included in the cell text

## [0.1.12] - 2024-07-09

### Added

- Add Generate Jupyter Notebook command

## [0.1.11] - 2024-07-08

### Fixed

- Fix explicit cell detection

## [0.1.10] - 2024-07-08

### Added

- Render Markdown within cells

## [0.1.9] - 2024-07-08

### Added

- New `evaluateImplicitCell`, bound to `Shift-Ctrl-Enter`, for evaluating the automatically detected cell even in the presence of explicit cell markers.
- The `evaluateCell` and `evaluateImplicitCell` commands will evaluate the current selection if present, instead of the current cell.

## [0.1.8] - 2024-07-08

### Fixed

- Fixed explicit cell detection

## [0.1.7] - 2024-07-08

### Added

- New `UseExplicitCellsIfPresent` option (default: false) controls whether to use `# %%` markers if found in the doc.

## [0.1.6] - 2024-07-08

### Fixed

- Release script fix

## [0.1.5] - 2024-07-08

### Added

- New `Evaluate Cell and Above` command

### Changed

- Use 'cell' consistenty (instead of 'block' or 'toplevel form')
- Use `colight.accentColor` theme color instead of configuration value

## [0.1.4] - 2024-07-05

### Changed

- Do not show current cell border by default

### Fixed

- Do not run extension when the python interactive window input is active

## [0.1.3] - 2024-07-05

### Added

- Show border above/below the current cell
- Use incremental parsing
- Allow configuration: debug logging, decorations
- Scroll next cell into view

## [0.1.2] - 2024-07-01

### Fixed

- Improved cell detection
- Tests passing

## [0.1.1] - 2024-06-30

### Added
- Initial release of Colight extension with core functionalities:
  - Cell detection (explicit and implicit)
  - Code execution in Jupyter
  - Visual feedback (cell highlighting and flash effect)
  - Navigation between cells
- Basic error handling and user feedback
- Configuration options for cell markers
- Initial test suite

