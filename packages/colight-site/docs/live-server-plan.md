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

### Phase 0: JSON-Based Content Delivery (NEW)

**Goal**: Transform LiveServer to deliver structured JSON instead of HTML, enabling dynamic client-side rendering

#### JSON Structure Design

Forms are ordered sequences of content that can contain interspersed markdown and code, with an optional visual at the end. Each form ends in either an expression, statement, or markdown:

```json
{
  "file": "example.py",
  "metadata": {
    "pragma": ["hide-statements"],
    "title": "Example Document"
  },
  "forms": [
    {
      "id": 0,
      "line": 1,
      "pragma": [],
      "content": [
        {
          "type": "markdown",
          "value": "## Introduction\nThis is markdown text from comments"
        }
      ],
      "endType": "markdown"
    },
    {
      "id": 1,
      "line": 5,
      "pragma": ["hide-code"],
      "content": [
        {
          "type": "markdown",
          "value": "Here we import our dependencies:"
        },
        {
          "type": "code",
          "value": "import numpy as np",
          "isStatement": true
        },
        {
          "type": "code",
          "value": "import pandas as pd",
          "isStatement": true
        }
      ],
      "endType": "statement"
    },
    {
      "id": 2,
      "line": 10,
      "pragma": [],
      "content": [
        {
          "type": "markdown",
          "value": "Let's create some data and visualize it:"
        },
        {
          "type": "code",
          "value": "x = np.linspace(0, 10, 100)",
          "isStatement": true
        },
        {
          "type": "code",
          "value": "y = np.sin(x)",
          "isStatement": true
        },
        {
          "type": "code",
          "value": "x, y",
          "isExpression": true
        },
        {
          "type": "visual",
          "format": "inline",
          "size": 1234,
          "data": "base64..."
        }
      ],
      "endType": "expression"
    },
    {
      "id": 3,
      "line": 15,
      "pragma": [],
      "content": [
        {
          "type": "code",
          "value": "print('Result computed')",
          "isStatement": true
        },
        {
          "type": "error",
          "value": "NameError: name 'result' is not defined"
        }
      ],
      "endType": "statement"
    }
  ]
}
```

#### Backend Tasks:

- [ ] Create `JsonFormGenerator` class that converts Forms to JSON
- [ ] Add `/api/document/<path>` endpoint that returns JSON document
- [ ] Preserve all form metadata including line numbers and pragma tags
- [ ] Maintain ordered content within each form (markdown/code interspersed)
- [ ] Track how each form ends (expression/statement/markdown)
- [ ] Include error information when forms fail to execute
- [ ] Handle visual output from expressions (inline or external)

#### Frontend Tasks:

- [ ] Update `live.jsx` to fetch JSON documents instead of HTML
- [ ] Create `FormRenderer` component that renders forms using colight APIs:
  - Use `md()` for markdown prose content
  - Use `tw()` for styling with Tailwind classes
  - Use existing syntax highlighting for code blocks
  - Reuse colight embed logic for visuals
- [ ] Implement visibility toggling based on pragma tags using CSS classes
- [ ] Add UI controls for toggling visibility (e.g., "Show/Hide Code")

#### Rendering Strategy:

```jsx
// Import colight APIs
import { tw, md, html } from "@colight/api";

// Content item renderer
function ContentRenderer({ item, pragma }) {
  const showCode =
    !pragma.includes("hide-code") &&
    !(pragma.includes("hide-statements") && item.isStatement);
  const showVisual = !pragma.includes("hide-visuals");
  const showProse = !pragma.includes("hide-prose");

  switch (item.type) {
    case "markdown":
      return showProse ? md({ className: "mb-4" }, item.value) : null;

    case "code":
      return showCode
        ? html([
            "pre",
            {
              className: tw("bg-gray-100 p-4 rounded-lg overflow-x-auto mb-4"),
            },
            ["code", { className: "language-python" }, item.value],
          ])
        : null;

    case "error":
      return html([
        "div.error",
        {
          className: tw(
            "bg-red-50 border border-red-200 text-red-700 p-4 rounded-lg mb-4",
          ),
        },
        ["pre", item.value],
      ]);

    case "visual":
      return showVisual
        ? html([
            "div.visual-container",
            { className: tw("mb-4") },
            // Will be populated by colight.loadVisuals()
            item.format === "inline"
              ? [
                  "script",
                  { type: "application/x-colight", "data-size": item.size },
                  item.data,
                ]
              : [
                  "div",
                  {
                    className: "colight-embed",
                    "data-src": item.path,
                    "data-size": item.size,
                  },
                ],
          ])
        : null;

    default:
      return null;
  }
}

// Form renderer
function FormRenderer({ form }) {
  // Skip empty forms
  if (!form.content || form.content.length === 0) return null;

  return html([
    "div.colight-form",
    {
      className: tw(`form-${form.id} ${form.pragma.join(" ")}`),
      "data-line": form.line,
      "data-form-id": form.id,
      "data-end-type": form.endType,
    },
    ...form.content.map((item) =>
      ContentRenderer({ item, pragma: form.pragma }),
    ),
  ]);
}

// Document renderer
function DocumentRenderer({ doc }) {
  return html([
    "div.colight-document",
    { className: tw("max-w-4xl mx-auto px-4 py-8") },
    ...doc.forms.map((form) => FormRenderer({ form })),
  ]);
}
```

#### Testing:

- JSON endpoint returns well-formed data
- All form types render correctly
- Pragma tags properly control visibility
- Markdown rendering works with all features
- Code syntax highlighting works
- Visuals load and display properly

#### Implementation Note:

Focus ONLY on shipping JSON and rendering it. Do not implement caching/diffing features yet. The goal is to:

1. Transform LiveServer to deliver JSON instead of HTML
2. Update the SPA to fetch and render JSON documents
3. Use colight's existing APIs (tw, md, html) for rendering
4. This change is ONLY for LiveServer, not for static builds

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

### Phase 8: Visuals Browser

**Goal**: Browse and manage generated .colight visualization files

#### Tasks:

- [ ] Add "Visuals" tab/section alongside "Files" in SPA
- [ ] Scan `.colight_cache` for generated `.colight` files
- [ ] Display visual thumbnails or metadata (size, creation time)
- [ ] Implement pruning logic for old .colight files:
  - Remove orphaned files (source .py deleted)
  - Clean files older than X days (configurable)
  - Keep recently accessed files
- [ ] Add visual file search/filter capabilities
- [ ] Quick preview on hover/click

#### API Endpoints:

- `/api/visuals` - List all .colight files with metadata
- `/api/visuals/prune` - Trigger cleanup of old files
- `/api/visuals/<path>` - Get specific visual data

#### Testing:

- Visuals tab shows all generated .colight files
- Pruning removes only appropriate files
- Visual previews work correctly
- Performance with many visual files

### Phase 9: Lazy Loading for Large Visuals

**Goal**: Optimize loading of large visualization files

#### Tasks:

- [ ] Modify generator to add `data-size` attribute to embed elements:
  - `<script type="application/x-colight" data-size="12345">...</script>`
  - `<div class="colight-embed" data-src="..." data-size="12345"></div>`
- [ ] Set size threshold for lazy loading (e.g., 100KB)
- [ ] Update embed.js to check data-size before loading
- [ ] Add placeholder with file size info for large visuals
- [ ] Implement click-to-load UI for large visuals:
  - Show size warning: "Large visualization (2.5MB) - click to load"
  - Loading progress indicator
  - Option to auto-load if user prefers
- [ ] Add user preference for auto-load threshold

#### Generator Changes:

```python
# In generator.py
if file_size < self.inline_threshold:
    # Inline with data-size
    lines.append(
        f'<script type="application/x-colight" data-size="{file_size}">\\n{base64_data}\\n</script>'
    )
else:
    # External with data-size
    lines.append(
        f'<div class="colight-embed" data-src="{embed_path}" data-size="{file_size}"></div>'
    )
```

#### Testing:

- Small visuals load immediately
- Large visuals show click-to-load UI
- data-size attributes are correctly added
- User preferences persist
- Loading indicators work smoothly

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
