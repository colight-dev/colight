# Output Panel Webview Migration Plan (React + Twind)

## Goals
- Replace the hand-written `media/output.js` and `media/output.css` with a React + Twind webview app.
- Reuse existing Colight JS utilities (Twind setup, base64 helpers, widget module APIs).
- Eliminate duplicate widgets and fragile DOM state handling.
- Keep the existing extension API surface intact (same message types and behaviors).
- bidirectional communication was not yet fully working, so we should try and figure out a robust design. the binary transit and general widget system have been working for +1 year but the "eval server" concept is new.

## Non-Goals
- No widget "revive" behavior across extension reloads (out of scope for this pass).

## Current Pain Points (Why this change)
- `media/output.js` duplicates logic already present in Colight JS code.
- Manual DOM mutation is brittle and creates duplicate widget instances.
- Fixes are hard to reason about and can regress core functionality (e.g. empty panel + eval timeout confusion).

## Proposed Architecture
- **New webview entrypoint**: `packages/colight-vscode/src/webview/OutputPanelApp.tsx`
  - React component tree with explicit state for widgets and panel mode.
  - Uses Twind (`packages/colight/src/js/utils.ts`) for styles.
  - Uses Colight base64 helpers (`packages/colight/src/js/base64.js`) to decode buffers.
  - Instead of the dynamic widget import `window.widgetModuleUri` this could be its own esbuild?


## Build/Bundle Strategy
- Extend `packages/colight-vscode/esbuild.js` to build two outputs:
  1. Extension bundle (`src/extension.ts`) -> `dist/extension.js` (current behavior).
  2. Webview bundle (`src/webview/OutputPanelApp.tsx`) -> `media/output.js`.
- Webview bundle config:
  - `platform: "browser"`, `format: "iife"` (or `esm` with `<script type="module">`).
  - `target: "es2020"` (or same as extension target).
  - `minify` in production mode.
  - include necessary widget utils

## Files to Add/Change
- Add:
  - `packages/colight-vscode/src/webview/OutputPanelApp.tsx`
  - (Optional) `packages/colight-vscode/src/webview/messages.ts` for message types.
- Update:
  - `packages/colight-vscode/esbuild.js` (add webview build step).
  - `packages/colight-vscode/src/outputPanel.ts` (point to new script, keep `window.widgetModuleUri`).
  - `packages/colight-vscode/media/output.css` (either remove or keep a tiny base file).
- Remove (after migration is working):
  - `packages/colight-vscode/media/output.js` (now generated).
  - Large hand-written CSS, if Twind covers all layout/styling.

## Implementation Steps (Detailed)
1. **Scaffold the React app**
   - Create `OutputPanelApp.tsx`.
   - Implement top-level layout: header, mode toggles, connection status, clear button.
   - Set up state: `mode`, `widgets[]`, `connectionState`, `errors`, `stdout`.

2. **Webview messaging**
   - Implement `window.addEventListener("message")`.
   - Handle all current message types and update React state accordingly.
   - Use a stable map of `evalId -> widget entry`.
   - On widget render completion, post `register-widget` back to the extension.

3. **Widget rendering**
   - Lazy-load widget module via `import(window.widgetModuleUri)`.
   - `parseColightData` + `render` into a ref container.
   - Inject `experimental` interface to enable widget commands.
   - Store `dispose()` in React state to clean up on unmount/remove.

4. **Update state handling**
   - On `update_state`, call `instance.updateWithBuffers(...)` for the relevant widget id.
   - Use Colight base64 helpers for buffer decode.

5. **Styling**
   - Use `tw(...)` for UI layout/styling to stay consistent with other Colight UI.
   - Keep a minimal `output.css` only for host-level defaults if needed.

6. **Bundling + wiring**
   - Update `esbuild.js` to emit `media/output.js`.
   - Update `outputPanel.ts` HTML to reference `media/output.js` (new bundle).

7. **Manual validation**
   - Snapshot mode renders exactly one widget and replaces on each eval.
   - Log mode prepends widgets without duplicates.
   - Close button removes widget on client + server.
   - `update_state` messages continue to apply.
   - Connection status flips when eval server stops/starts.

## Acceptance Criteria
- No duplicate widgets in Snapshot mode.
- Remove/clear always removes all related widget instances.
- Webview shows "No Output Yet" only when there are zero widgets.
- All functionality currently supported by `media/output.js` is preserved.
- New webview bundle is produced by `node esbuild.js`.

## Risks / Mitigations
- **Risk**: Webview CSP issues with module scripts.
  - **Mitigation**: Use `iife` bundle or add `type="module"` with CSP update.
- **Risk**: Twind global setup injects styles multiple times.
  - **Mitigation**: Use existing `utils.ts` guard with `__TWIND_INSTANCE__`.
- **Risk**: Widget renderer module import fails.
  - **Mitigation**: Keep runtime error UI and log to console.
