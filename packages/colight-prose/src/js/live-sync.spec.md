# Live Sync Specification

## Overview

This specification describes the live sync behavior between the colight-prose server and client. The implementation is located in:

- **Client**: `packages/colight-prose/src/js/live.jsx` (main app), `websocket-message-handler.js` (message processing), `TopBar.jsx` (UI components)
- **Server**: `packages/colight-prose/src/colight_prose/server.py` (WebSocket server and file watching)

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

1. **Multi-Client Coordination**: The server broadcasts to all clients but has no concept of individual client state. This could lead to confusing behavior if multiple clients have different pinned files.

2. **Race Conditions**: If a user navigates while a file update is in progress, there could be race conditions between the navigation request and incoming updates.

3. **Memory Management**: The `blockResults` state accumulates all received blocks but there's no cleanup for files that are no longer being viewed.

4. **WebSocket Reconnection**: When the WebSocket reconnects, the client doesn't automatically re-request the current file, potentially leaving stale content displayed.

5. **Directory Changes**: The system watches for file changes but doesn't update the directory tree when files are added/removed.

6. **Performance**: Every file change triggers a full execution cycle. For large files or many simultaneous changes, this could impact performance.

7. **Error States**: Limited error feedback to users when file processing fails or WebSocket connection issues occur.

## ~~Next Patch~~ IMPLEMENTED: Client-Aware Incremental Execution ✅

### Problem Statement

~~The current implementation has~~ HAD significant inefficiencies:

1. **Over-processing** ✅: ~~Server processes ALL files on every change~~
   - **FIXED**: Server now only processes files being watched by clients
2. **Over-broadcasting** ✅: ~~Server sends updates for all changed files to all clients~~
   - **FIXED**: Server sends updates only to clients watching specific files
3. **Memory waste** ⏳: Clients maintain state for files they're not viewing
   - **TODO**: Phase 4 will implement client-side state scoping
4. **Cache growth** ✅: ~~Server cache grows unbounded~~
   - **FIXED**: Cache manager evicts entries for unwatched files
5. **Resource waste** ✅: ~~Server spends cycles checking blocks in unwatched files~~
   - **FIXED**: Execution filtered by watched files

### Proposed Solution

Implement client-aware execution and targeted updates:

1. **Client Registration**: Clients explicitly register which files they're watching
2. **Dependency-Aware Execution**: Server only evaluates watched files and their dependencies
3. **Targeted Updates**: Server only sends updates to clients watching the affected files
4. **Scoped Client State**: Clients only maintain state for their currently viewed file

### Implementation Details

#### New Client → Server Messages ✅ IMPLEMENTED

1. **`watch-file`**: Register interest in a file

   ```json
   {
     "type": "watch-file",
     "path": "path/to/file.py",
     "clientId": "unique-client-id"
   }
   ```

2. **`unwatch-file`**: Unregister interest in a file
   ```json
   {
     "type": "unwatch-file",
     "path": "path/to/file.py",
     "clientId": "unique-client-id"
   }
   ```

#### New Server → Client Messages ✅ IMPLEMENTED

1. **`file-changed`**: Notification when a single file changes
   ```json
   {
     "type": "file-changed",
     "path": "path/to/file.py",
     "watched": true // Whether any client is watching this file
   }
   ```
   - Sent to ALL clients (not just watchers)
   - Only sent when exactly one file changes in the throttle window (100ms)
   - Clients decide whether to navigate based on pin state
   - Enables "follow mode" for unpinned clients

#### Server State Changes ✅ IMPLEMENTED

1. **Client Registry**:

   ```python
   class ClientRegistry:
       def __init__(self):
           self.clients: Dict[str, WebSocket] = {}  # clientId -> WebSocket
           self.watched_files: Dict[str, Set[str]] = {}  # file_path -> Set[clientId]
           self.client_files: Dict[str, str] = {}  # clientId -> current_file
   ```

2. **Dependency Tracking** ✅:

   - Custom AST-based import analysis implemented
   - FileDependencyGraph class tracks forward and reverse dependencies
   - Handles relative and absolute imports, filters external dependencies
   - Caches results with mtime-based invalidation

3. **Execution Strategy with Cache Management**:

   ```python
   def on_file_change(changed_file):
       # Find all files that depend on changed_file
       affected_files = dependency_graph.get_affected(changed_file)

       # Filter to only watched files
       files_to_execute = {
           f for f in affected_files
           if f in self.watched_files
       }

       # Execute only what's needed
       for file in files_to_execute:
           results = execute_file(file)  # Uses block-level cache

           # Send only to watching clients
           for client_id in self.watched_files[file]:
               send_to_client(client_id, results)

       # Mark unwatched affected files for cache eviction
       unwatched_affected = affected_files - files_to_execute
       block_cache.mark_for_eviction(unwatched_affected)
   ```

4. **Cache Management** ✅:
   - BlockCache tracks entries by source file
   - Automatic eviction for unwatched files (30-second intervals)
   - LRU eviction with 500MB default limit
   - Hot entry protection based on access count and recency

#### Client State Changes

1. **Single-File State**:

   - Remove the accumulating `blockResults` for all files
   - Only store results for the currently viewed file
   - Clear state when navigating away

2. **Navigation Cleanup**:

   ```javascript
   const navigateTo = (newFile) => {
     // Unwatch previous file
     if (currentFile) {
       ws.send({ type: "unwatch-file", path: currentFile, clientId });
     }

     // Clear old state
     setBlockResults({});

     // Watch new file
     ws.send({ type: "watch-file", path: newFile, clientId });

     // Navigate
     navigate(newFile);
   };
   ```

### Open Questions / TBD

1. **Dependency Detection**:

   - Should we use existing Python reload libraries (e.g., `watchdog`, `jurigged`)?
   - How deep should dependency tracking go? (stdlib imports? third-party?)
   - Cache dependency graph or recompute on each change?

2. **Client Identity**:

   - How to generate stable client IDs?
   - Handle reconnection with same ID?
   - Clean up state for disconnected clients?

3. **Multi-Client Scenarios**:

   - If multiple clients watch the same file, execute once and broadcast to all?
   - How to handle different clients with different pragma settings?

4. **Performance Considerations**:

   - Lazy dependency resolution vs upfront graph building?
   - Should we batch file changes within a time window?

5. **Backwards Compatibility**:
   - Support clients that don't send watch/unwatch messages?
   - Graceful degradation to current broadcast-all behavior?

### Benefits

1. **Reduced Server Load**: Only execute what clients need
2. **Reduced Network Traffic**: Targeted updates instead of broadcast
3. **Better Client Performance**: Less state to manage, fewer updates to process
4. **Scalability**: Can support more clients since work scales with watched files, not total files

### Migration Strategy

1. **Phase 1**: Add watch/unwatch protocol without changing execution (measure usage patterns)
2. **Phase 2**: Implement dependency tracking (but still execute everything)
3. **Phase 3**: Switch to targeted execution with feature flag
4. **Phase 4**: Client-side state scoping
5. **Phase 5**: Remove legacy broadcast behavior

### Related Improvements

Consider leveraging Python auto-reload libraries:

- **watchdog**: File system monitoring
- **jurigged**: Live code reloading with dependency tracking
- **reloadr**: Minimal reload with import tracking

These libraries already solve the dependency tracking problem and could provide a solid foundation.
