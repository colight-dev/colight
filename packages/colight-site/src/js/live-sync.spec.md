# Live Sync Specification

## Overview

This specification describes the live sync behavior between the colight-site server and client. The implementation is located in:

- **Client**: `packages/colight-site/src/js/live.jsx` (main app), `websocket-message-handler.js` (message processing), `TopBar.jsx` (UI components)
- **Server**: `packages/colight-site/src/colight_live/server.py` (WebSocket server and file watching)

## Core Principles

The live sync system enables real-time updates between the server (watching files) and client (displaying content). The system should balance automatic navigation with user control through pinning.

## Desired Behavior

### Axioms / Invariants

1. **Single Source of Truth**: The server is the single source of truth for file content and changes
2. **Client Autonomy**: The client decides navigation behavior based on its own state (pinned file, current file)
3. **Lossless Updates**: All file changes are processed and available, regardless of navigation decisions
4. **Explicit User Intent**: Pinning represents explicit user intent to stay on a file, which overrides automatic navigation

### Navigation Rules

1. **When NOT pinned**:

   - If another file changes, automatically navigate to that file
   - Show the latest changes immediately

2. **When pinned**:

   - Stay on the pinned file regardless of other file changes
   - The pinned file still receives its own updates when it changes
   - Other files' updates are processed in the background but don't cause navigation

3. **Pin State Visibility**:
   - A pin emoji (📌) appears next to the filename in the breadcrumb when pinned
   - The filename button has a blue background when pinned
   - Hovering shows tooltip text indicating pin state
   - Users can toggle pin state by clicking the filename in breadcrumb or using command bar (Cmd/Ctrl+K)

## State Management

### Client State

1. **Navigation State**:

   - `currentFile`: The file currently being viewed
   - `currentPath`: The full path including directories
   - `isDirectory`: Whether viewing a directory or file

2. **Pinning State**:

   - `pinnedFile`: The file path that is pinned (null if nothing pinned)

3. **Content State**:

   - `blockResults`: Current block execution results for the displayed file
   - `latestRun`: Version number of the latest run from server

4. **UI State**:
   - `pragmaOverrides`: User preferences for hiding/showing content types
   - `directoryTree`: Cached directory structure for navigation

### Server State

1. **File Monitoring**:

   - Active file watchers for the configured paths
   - File change detection and debouncing

2. **Execution State**:

   - `IncrementalExecutor`: Maintains execution context and caching
   - Block dependency graph and execution order
   - Visual data store for generated visualizations

3. **Client Tracking**:
   - Active WebSocket connections
   - No per-client state (stateless with respect to individual clients)
   - Run version counter (monotonic, shared across all clients)

### What Server Knows About Clients

The server maintains minimal client knowledge:

- Active WebSocket connections (for broadcasting)
- Client's requested file and run version (via `request-load` messages)
- No persistent client state or preferences

## Message Flow

### Client → Server

1. **`request-load`**: Client requests a specific file
   ```json
   {
     "type": "request-load",
     "path": "path/to/file.py",
     "clientRun": 123
   }
   ```

### Server → Client

1. **`run-start`**: Indicates execution beginning

   ```json
   {
     "type": "run-start",
     "file": "path/to/file.py",
     "run": 124,
     "blocks": ["block1", "block2"],
     "dirty": ["block1"]
   }
   ```

   Note: File paths now include the `.py` extension

2. **`block-result`**: Individual block execution results

   ```json
   {
     "type": "block-result",
     "block": "block1",
     "run": 124,
     "elements": [...],
     "ok": true
   }
   ```

3. **`run-end`**: Execution completed
   ```json
   {
     "type": "run-end",
     "run": 124,
     "error": null
   }
   ```

## Decision Making

### Client Decisions

1. **Navigation**: Whether to navigate to a changed file based on:

   - Current pinned state
   - Whether the changed file is already being viewed
   - User interactions (clicking files, using command bar)

2. **Content Display**: What to show/hide based on pragma overrides

3. **Update Handling**: How to merge incoming updates with existing state

### Server Decisions

1. **File Processing**: Which files to process based on include/ignore patterns
2. **Execution Order**: Block execution order based on dependencies
3. **Caching**: Whether to use cached results or re-execute blocks
4. **Broadcasting**: When to send updates to all connected clients

## Expected Behaviors

1. **File Change Detection**:

   - Server detects file change
   - Server executes changed blocks and their dependents
   - Server broadcasts updates to all clients
   - Each client decides independently whether to navigate

2. **Manual Navigation**:

   - User clicks a file or uses command bar
   - Client sends `request-load` to server
   - Server processes and sends full file state
   - Client displays the requested file

3. **Pinning Toggle**:
   - User clicks file name in breadcrumb or uses command bar
   - Pin state updates immediately in UI
   - Subsequent file changes respect pin state

## Things I Noticed

1. **Pin State Visibility**: The pinned state is only visible when hovering over the file name in the breadcrumb (via blue background). There's no persistent visual indicator that a file is pinned.

2. **Multi-Client Coordination**: The server broadcasts to all clients but has no concept of individual client state. This could lead to confusing behavior if multiple clients have different pinned files.

3. **Race Conditions**: If a user navigates while a file update is in progress, there could be race conditions between the navigation request and incoming updates.

4. **Memory Management**: The `blockResults` state accumulates all received blocks but there's no cleanup for files that are no longer being viewed.

5. **WebSocket Reconnection**: When the WebSocket reconnects, the client doesn't automatically re-request the current file, potentially leaving stale content displayed.

6. **Directory Changes**: The system watches for file changes but doesn't update the directory tree when files are added/removed.

7. **Performance**: Every file change triggers a full execution cycle. For large files or many simultaneous changes, this could impact performance.

8. **Error States**: Limited error feedback to users when file processing fails or WebSocket connection issues occur.

## Next Patch: [To be determined]
