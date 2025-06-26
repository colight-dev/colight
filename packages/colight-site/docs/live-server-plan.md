# LiveServer Implementation Plan

## Overview

LiveServer is an on-demand development server for colight-site that builds files only when accessed, providing efficient development workflows for large projects. Unlike `watch-serve` which pre-builds everything, LiveServer generates content on-demand and tracks file activity for smart navigation.

## Key Design Decisions

1. **Name**: `LiveServer` (not "serve") to distinguish from static serving
2. **Build Cache**: Mirror source directory structure in `.colight_cache`
3. **Batch Changes**: Skip (ignore) rapid multi-file changes to avoid noise
4. **Purpose**: Development-only tool; `build` command remains for publishing

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   File System   │────▶│   LiveServer     │────▶│    Browser      │
│   (*.py files)  │     │                  │     │                 │
└─────────────────┘     │ - FileTracker    │     └─────────────────┘
                        │ - OnDemandBuilder │              ▲
                        │ - WebSocket       │              │
                        │ - HTTP Server     │              │
                        └──────────────────┘              │
                                 │                         │
                                 ▼                         │
                        ┌──────────────────┐              │
                        │  .colight_cache  │──────────────┘
                        │ (mirrors source) │
                        └──────────────────┘
```

## Implementation Phases

### Phase 1: Core Infrastructure ✓

**Goal**: Basic on-demand server that builds files when accessed

#### Tasks:

- [x] Create `live_server.py` module
- [x] Implement `LiveServer` class with basic HTTP server
- [x] Add `OnDemandMiddleware` to intercept .html requests
- [x] Implement on-demand build logic
- [x] Ensure `.colight_cache` mirrors source directory structure
- [x] Add basic CLI command `live` to `cli.py`

#### Testing:

- Start server with a simple .py file
- Access the file via browser
- Verify it builds on first access
- Verify cached version is served on subsequent access

### Phase 1.5: SPA Frontend

**Goal**: Transform LiveServer into a modern SPA with smart navigation

#### Tasks:

- [ ] Transform `live.jsx` into full SPA with client-side routing
- [ ] Add API endpoints to LiveServer:
  - `/api/files` - Returns JSON list of all .py files
  - `/api/content/<path>` - Returns HTML content for a file
- [ ] Implement fuzzy file search with Cmd/Ctrl+K
- [ ] Add minimal topbar with:
  - Current file path
  - Pin/unpin button
  - Connection status indicator
- [ ] Update WebSocket protocol:
  - Send `{type: 'file-changed', path: '/path/to/file.html'}` instead of generic reload
  - Send `{type: 'files-updated'}` when file list changes
- [ ] Implement auto-navigation:
  - Navigate to changed file automatically
  - Unless current file is pinned
  - Skip navigation for batch changes
- [ ] Remove HtmlFallbackMiddleware for SPA mode
- [ ] Serve SPA for all non-API routes

#### Architecture:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   File System   │────▶│   LiveServer     │────▶│    SPA (React)  │
│   (*.py files)  │     │                  │     │                 │
└─────────────────┘     │ - API Endpoints  │     │ - Routing       │
                        │ - OnDemandBuilder │     │ - File Search   │
                        │ - WebSocket       │     │ - Content View  │
                        └──────────────────┘     │ - Pin/Navigate  │
                                 │               └─────────────────┘
                                 ▼                         ▲
                        ┌──────────────────┐              │
                        │  .colight_cache  │              │
                        │ (API responses)  │──────────────┘
                        └──────────────────┘
```

#### Testing:

- File search works with fuzzy matching
- Auto-navigation works for single file changes
- Pinned files don't auto-navigate
- API endpoints return correct data
- SPA loads for all routes

### Phase 2: File Tracking System

**Goal**: Track file metadata and detect changes

#### Tasks:

- [ ] Create `FileTracker` class in `live_server.py`
- [ ] Track file modification times
- [ ] Implement recently edited files list (in-memory)
- [ ] Add batch change detection and skipping logic
- [ ] Integrate file watcher to update metadata

#### Testing:

- Edit a file and verify timestamp updates
- Make batch changes and verify they're skipped
- Check recently edited list maintains correct order

### Phase 3: Enhanced File Tracking API

**Goal**: Provide rich metadata through API for SPA

#### Tasks:

- [ ] Enhance `/api/files` endpoint to include:
  - File modification times
  - Build status (built/not-built)
  - Relative timestamps
  - File sizes
- [ ] Add `/api/recent` endpoint for recently edited files
- [ ] Include file count and last activity in responses
- [ ] Sort files by recent activity option

#### Testing:

- API returns comprehensive file metadata
- Recently edited files API works correctly
- Build status indicators are accurate

### Phase 4: Smart Navigation (Simplified)

**Goal**: Enhanced navigation features in SPA

#### Tasks:

- [ ] Batch change detection in FileTracker
- [ ] Add navigation preferences to SPA
- [ ] Keyboard navigation (arrow keys in search)
- [ ] History tracking (back/forward)
- [ ] URL state preservation

#### Testing:

- Batch changes don't trigger navigation
- Keyboard navigation works smoothly
- Browser back/forward works correctly

### Phase 5: Selective Content Updates

**Goal**: Efficient content updates without page reloads

#### Tasks:

- [ ] Implement content refresh in SPA
- [ ] Only update content if viewing changed file
- [ ] Live update file list when files added/removed
- [ ] Smooth transitions between content updates
- [ ] Show update indicator when content refreshes

#### Testing:

- Change file A while viewing file B (no update)
- Change file A while viewing file A (content updates)
- File list updates when files are added/removed

### Phase 6: Performance Optimization

**Goal**: Optimize for large projects

#### Tasks:

- [ ] Implement build cache with proper invalidation
- [ ] Add memory limits for file tracking
- [ ] Optimize index generation for many files
- [ ] Add progress indicators for slow builds
- [ ] Implement cleanup for old cache files

#### Testing:

- Test with 100+ files
- Verify memory usage stays reasonable
- Check cache invalidation works correctly
- Measure build times and optimize

### Phase 7: Developer Experience

**Goal**: Polish and usability improvements

#### Tasks:

- [ ] Add build error display in browser
- [ ] Implement keyboard shortcuts (r=refresh, i=index)
- [ ] Add build time indicators
- [ ] Create status bar with server info
- [ ] Add --open-last flag to open last edited file

#### Testing:

- Verify error display works
- Test all keyboard shortcuts
- Check status indicators update correctly

## File Structure

```
packages/colight-site/src/colight_site/
├── live_server.py       # Main LiveServer implementation with API endpoints
├── cli.py               # Modified: Add 'live' command
└── watcher.py          # Reference: Reuse watch logic

src/js/
└── live.jsx            # SPA frontend application

dist/
└── live.js             # Built SPA bundle

.colight_cache/         # Build output (mirrors source)
├── examples/
│   ├── basic.html      # On-demand generated content
│   └── advanced.html
└── (no index.html - SPA handles file listing)
```

## CLI Interface

```bash
# Start LiveServer
colight-site live [input_path] [options]

# Options:
--port/-p          Port for HTTP server (default: 5500)
--host/-h          Host to bind (default: 127.0.0.1)
--no-open          Don't open browser on start
--open-last        Open last edited file on start
--verbose/-v       Verbose output
--include          File patterns to include
--ignore           File patterns to ignore
```

## Success Criteria

1. **Performance**: Instant startup, even with 1000+ files
2. **Efficiency**: Only builds requested files
3. **Smart**: Tracks activity and provides intelligent navigation
4. **Reliable**: Handles edge cases (errors, missing files, etc.)
5. **Developer-friendly**: Clear feedback and helpful features

## Notes

- LiveServer is for development only; production builds use existing `build` command
- All built files go in `.colight_cache` with proper directory structure
- Batch changes (e.g., git operations) are intelligently ignored
- Focus on developer experience and fast iteration cycles
- **Static build compatibility**: The `build` command generates static HTML that looks identical to LiveServer preview (without live chrome)
- **No backwards compatibility needed**: This is greenfield development, prioritize simplicity
- **Single build artifact**: One `live.jsx` handles all frontend functionality
