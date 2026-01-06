# Colight Development Reference

## Build/Run Commands

- Build frontend: `yarn build`
- Watch mode: `yarn dev`
- Run tests: `yarn test` (JS and Python)
- Run JS tests only (watch mode): `yarn test:js`
- Run single JS test: `yarn vitest <test-file-pattern>`
- Run specific JS test file: `yarn vitest run packages/colight/tests/js/some-test.js`
- Run Python tests: `yarn test:py` or `uv run pytest packages/colight/tests/`
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

- In Python notebooks, use our own block structure - comments are markdown, expressions will have their return value visualized. No jupytext cell markers are necessary. Empty lines separate blocks.
- A Colight usage guide for LLMs is in `docs/llms.py`.
- When writing React components, use Tailwind classes, wrapping in `tw` from `src/js/utils.ts`.
- We control the whole Colight stack - server and client: design with this in mind. Don't "work around" issues that come from another layer in the stack: solve the problem at its origin, aim for the cleanest end-to-end solution.
- This software is not yet released, so we don't need to maintain backwards compatibility or keep around legacy code.

For detailed patterns, review existing code in the corresponding module.

## Monorepo Structure

This project uses a monorepo structure with the main package:

- `packages/colight/` - Main visualization library with runtime, publish, and server modules
- Root workspace manages shared dependencies and tooling

When working on the package, navigate to the package directory or use the workspace commands from the root.

## Development Guidelines

- Use yarn, not npm
- If introducing a change which may increase complexity (eg. as a workaround for some difficulty), always ask the user first, they may have an idea for a simpler solution.
- **Tests**: Never put test files at the root level. Always place tests in the `tests` directory:
  - `packages/colight/tests/` for all colight tests

## Testing Best Practices

- **Test Real Code**: Always test actual exported functions/components, never create "mirror" implementations that duplicate logic (they will drift from reality)
- **File Naming**: Avoid naming non-test files with `test_` prefix in Python directories - pytest will try to import them
- **Component Testing**: When components need testing, export them properly (e.g., `export { ComponentName }` or `export default ComponentName`)
- **React Component Tests**: Simple smoke tests are often sufficient - check rendering, basic interactions, and prop handling
- **React Navigation Testing**: When testing navigation with React Router:
  - Use `MemoryRouter` with `initialEntries` to set routes
  - For testing route changes, use `rerender()` with a new `MemoryRouter` instead of `window.history` methods
  - Example:
    ```javascript
    const { rerender } = render(
      <MemoryRouter initialEntries={["/file1.py"]}>
        <Routes>
          <Route path="*" element={<App />} />
        </Routes>
      </MemoryRouter>,
    );
    // Navigate by re-rendering with new route
    rerender(
      <MemoryRouter initialEntries={["/file2.py"]}>
        <Routes>
          <Route path="*" element={<App />} />
        </Routes>
      </MemoryRouter>,
    );
    ```
- **Vitest Configuration**: The `vitest.config.mjs` includes all colight tests:
  ```javascript
  include: [
    "packages/colight/tests/**/*.test.{js,mjs,cjs,ts,mts,cts,jsx,tsx}",
  ];
  ```
