# Colight Development Reference

## Build/Run Commands

- Build frontend: `yarn build`
- Watch mode: `yarn dev`
- Run tests: `yarn test` (JS, Python, and colight-site)
- Run JS tests only (watch mode): `yarn test:js`
- Run single JS test: `yarn vitest <test-file-pattern>`
- Run specific JS test file: `yarn vitest run packages/colight/tests/js/some-test.js`
- Run Python tests: `yarn test:py` or `uv run pytest packages/colight/tests/`
- Run colight-site tests: `yarn test:site` or `uv run pytest packages/colight-site/tests/`
- Run single Python test: `uv run pytest packages/colight/tests/test_file.py::test_function`
- **IMPORTANT**: Always use `uv run python` instead of `python` directly
- Typecheck Python: `pyright` or `yarn pyright`
- Format & lint: `pre-commit run --all-files`
- Docs: `yarn docs:watch` to serve, `yarn docs:build` to build

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
- `packages/colight-site/` - Static site generator for .py files
- Root workspace manages shared dependencies and tooling

When working on specific packages, navigate to the package directory or use the workspace commands from the root.

## Development Guidelines

- Use yarn, not npm
- If introducing a change which may increase complexity (eg. as a workaround for some difficulty), always ask the user first, they may have an idea for a simpler solution.
- **Tests**: Never put test files at the root level. Always place tests in a `tests` directory within the appropriate package. For example:
  - `packages/colight/tests/` for colight package tests
  - `packages/colight-site/tests/` for colight-site package tests

## Testing Best Practices

- **Test Real Code**: Always test actual exported functions/components, never create "mirror" implementations that duplicate logic (they will drift from reality)
- **File Naming**: Avoid naming non-test files with `test_` prefix in Python directories - pytest will try to import them
- **Component Testing**: When components need testing, export them properly (e.g., `export { ComponentName }` or `export default ComponentName`)
- **React Component Tests**: Simple smoke tests are often sufficient - check rendering, basic interactions, and prop handling
- **Vitest Configuration**: In the monorepo, `vitest.config.mjs` must include both package test paths:
  ```javascript
  include: [
    "packages/colight/tests/**/*.test.{js,mjs,cjs,ts,mts,cts,jsx,tsx}",
    "packages/colight-site/tests/**/*.test.{js,mjs,cjs,ts,mts,cts,jsx,tsx}",
  ];
  ```
