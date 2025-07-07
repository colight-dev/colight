# Colight Live: Implementation Plan

## 1. Executive Summary

This document outlines the implementation plan for **Colight Live**, a development environment that provides near real-time feedback for `.colight.py` files. As a user edits their file, a browser view will intelligently update to reflect the changes.

Our core design philosophy is **edit-time reactivity**. The system reacts to _code changes_, not runtime data changes (e.g., from UI widgets). This provides a simple and predictable mental model for the user: "the file runs from top to bottom," while our implementation uses intelligent caching and dependency analysis to make this process extremely fast.

## 2. Core Architecture

The system is triggered by file changes and orchestrates a flow from parsing to execution to browser updates.

1.  **File Watcher**: Monitors the target `.colight.py` file for save events. This is the entry point for the update cycle.
2.  **Form Parser**: Divides the Python script into executable blocks called "forms."
    - **Starting Point**: We will build upon the existing `colight-site` implementation, which uses `libcst` to parse scripts into these computational units.
3.  **Static Dependency Analyzer**: Uses AST (Abstract Syntax Tree) analysis to build a dependency graph between forms. It determines which forms define ("write") variables and which forms use ("read") them.
4.  **Execution Engine**: Runs the Python forms in order. It maintains an in-memory cache of results (including large objects like DataFrames) to avoid re-computing unchanged forms.
5.  **WebSocket Server/Client**: Manages the communication channel to the browser, sending granular updates.

## 3. Key Design Decisions

Our design prioritizes simplicity, correctness, and a predictable user experience.

- **Dependency Analysis: Static-Only**
  We will rely exclusively on static AST-based analysis to determine dependencies. We will not use any runtime analysis or execution wrapping. This keeps the implementation simple and avoids unexpected behavior. The analysis will track variable assignments (writes) and their usage (reads) to build the graph.

- **Form Identity: Content-Based Hashing**
  A form's identity will be determined by a hash of its content. When a form is edited, its hash changes. The system will treat the edited form as a _replacement_ for the old one at the same position in the file. Advanced similarity-based tracking is out of scope as it adds unnecessary complexity. The dependency graph must be ableto handle forms being removed and replaced elegantly.

- **Execution Model: Incremental with In-Memory Cache**
  When a form changes, the engine will:

  1.  Perform a fresh static analysis on the changed form.
  2.  Identify all dependent forms through the dependency graph (i.e., any form that reads a variable the changed form writes, plus all of their dependents).
  3.  Re-execute the changed form and all its dependents in the correct topological order.
  4.  Results of unchanged forms will be served directly from the in-memory cache, preserving Python objects without serialization.

- **Side Effects: Minimal Handling**
  We will not attempt to comprehensively manage all imperative side effects. The philosophy is to encourage best practices.
  - The one exception is **Matplotlib's global state**, which will be cleared between form executions to prevent plots from bleeding into each other.
  - Other side effects (e.g., file I/O, database writes) will be left to the user to manage. We will provide clear documentation on best practices for working with Colight Live.

## 4. Implementation Phases

### Phase 1: Core Execution & Caching

- **Goal**: Prove the core caching and re-execution logic.
- **Tasks**:
  1.  Implement the **File Watcher** to trigger updates on save.
  2.  Integrate the existing `colight-site` **Form Parser**.
  3.  Build the **Static Dependency Analyzer** to track variable reads/writes.
  4.  Develop the **In-memory Execution Engine** that can execute a list of forms and skip unchanged forms based on their content hash.
  5.  Set up a basic **WebSocket server** that, upon any change, re-runs the necessary forms and sends the _entire_ resulting HTML to the browser for a full refresh.

### Phase 2: Granular Updates & Graph Management

- **Goal**: Achieve a smooth, flicker-free browser experience.
- **Tasks**:
  1.  Refine the dependency graph to ensure it can handle forms being replaced, inserted, and deleted correctly.
  2.  Define and implement a **granular WebSocket protocol** with messages like `form-update`, `form-insert`, `form-delete`.
  3.  Build the browser client to handle these specific messages, using a library like `morphdom` to patch the DOM without a full page reload.
  4.  Improve the display of syntax or runtime errors, showing them inline with the form that caused them.

### Phase 3: Robustness & Side Effects

- **Goal**: Handle real-world code gracefully.
- **Tasks**:
  1.  Implement the **Matplotlib state-clearing** logic that runs between form executions.
  2.  Add static analysis detectors for a minimal set of "low-hanging fruit" patterns that might cause issues (e.g., `open()` outside a `with` block) and display clear, non-intrusive warnings to the user.
  3.  Improve overall error handling and reporting.

### Phase 4: Polish & User Experience

- **Goal**: Create a polished, production-ready tool.
- **Tasks**:
  1.  Implement user-facing "escape hatches" for controlling execution, such as a `# colight: no-cache` comment to force a form to always re-run.
  2.  Write comprehensive user documentation explaining the execution model and best practices.
  3.  Refine the UI/UX for loading states, errors, and warnings.

## 5. Out of Scope

To maintain simplicity and focus, the following features are explicitly out of scope for the initial implementation:

- **Runtime Dependency Tracking**: All analysis will be static.
- **Advanced Form Identity Tracking**: No content-similarity or fuzzy matching.
- **Automatic Side-Effect Management**: Beyond Matplotlib, users are responsible for managing side effects.
- **Parallel Execution**: Execution will be sequential.
