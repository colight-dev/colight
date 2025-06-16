# Colight Development Reference

## Build/Run Commands

- Build frontend: `yarn build`
- Watch mode: `yarn dev`
- Run tests: `yarn test` (both JS and Python)
- Run JS tests only (watch mode): `yarn test:js`
- Run single JS test: `yarn vitest <test-file-pattern>`
- Run Python tests: `yarn test:py` or `uv run pytest tests/python/`
- Run single Python test: `uv run pytest tests/python/test_file.py::test_function`
- Typecheck Python: `pyright` or `yarn pyright`
- Format & lint: `pre-commit run --all-files`
- Docs: `yarn watch:docs` to serve, `yarn build:docs` to build

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
