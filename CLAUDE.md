# Colight Development Reference

## Build/Run Commands

- Build frontend: `yarn build`
- Watch mode: `yarn dev`
- Run tests: `yarn test` (JS, Python, and colight-site)
- Run JS tests only (watch mode): `yarn test:js`
- Run single JS test: `yarn vitest <test-file-pattern>`
- Run Python tests: `yarn test:py` or `uv run pytest packages/colight/tests/`
- Run colight-site tests: `yarn test:colight-site` or `uv run pytest packages/colight-site/tests/`
- Run single Python test: `uv run pytest packages/colight/tests/test_file.py::test_function`
- **IMPORTANT**: Always use `uv run python` instead of `python` directly
- Typecheck Python: `pyright` or `yarn pyright`
- Format & lint: `pre-commit run --all-files`
- Docs: `yarn watch:docs` to serve, `yarn build:docs` to build

## Visual Testing

- Run visual regression tests: `uv run pytest tests/visual/ -v`
- Update visual baselines: `uv run python scripts/update_visual_baselines.py`
- Visual tests run in CI and compare pixel-perfect against baselines
- Update baselines when you intentionally change visual output

## Code Style Guide

- **Python**: snake_case for variables/functions, PascalCase for classes
- **JS/TS**: camelCase for variables/functions, PascalCase for components/classes
- **Imports**: stdlib first, third-party next, then project-specific (alphabetical)
- **Types**: Use type hints everywhere in Python, TypeScript for JS
- **Documentation**: Google-style docstrings with Args/Returns sections
- **Error Handling**: Descriptive error messages, prefer specific exceptions
- **Testing**: Use pytest for Python, Vitest for JS/TS
- **Formatting**: Enforced by ruff-format (Python) and pre-commit hooks

## Conventions / Approach

- In Python notebooks, use Jupytext cell boundaries.
- A Colight usage guide for LLMs is in `docs/llms.py`.
- When writing React components, use Tailwind classes, wrapping in `tw` from `src/js/utils.ts`.

For detailed patterns, review existing code in the corresponding module.

## Monorepo Structure

This project uses a monorepo structure with multiple packages:

- `packages/colight/` - Main visualization library
- `packages/colight-site/` - Static site generator for .colight.py files
- Root workspace manages shared dependencies and tooling

When working on specific packages, navigate to the package directory or use the workspace commands from the root.

## Development Guidelines

- Use yarn, not npm
