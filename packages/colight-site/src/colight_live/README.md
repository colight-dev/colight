# `colight_live` Architecture

This document provides a high-level overview of the `colight_live` development server's architecture for future developers.

## Overview

`colight_live` is a sophisticated live-reloading development server for `.py` documents. Its primary goal is to provide a fast and interactive development experience by intelligently re-executing only the necessary parts of a document when code changes are made. It combines a web server, a WebSocket server, and a multi-layered dependency analysis and caching system to achieve this.

## System Workflow

Understanding the flow of events from a file change to a UI update is key to understanding the system.

1.  **File Change Detection**: The `LiveServer` in `server.py` uses the `watchfiles` library to monitor the filesystem for changes to any `.py` files in the target directory.

2.  **File-Level Dependency Analysis**: When a file is changed, the `FileDependencyGraph` (`file_graph.py`) is consulted. This graph, built by analyzing `import` statements, determines the "blast radius" of the change—i.e., all other files that import the changed file, directly or indirectly.

3.  **Client Notification (Run Start)**: For each file that is affected and currently being watched by a client, the server sends a `run-start` message via WebSocket. This tells the frontend to prepare for incoming results.

4.  **Block-Level Dependency Analysis**: The server then parses the changed file into a series of code and prose blocks.

    - The `DependencyAnalyzer` (`dependency_analyzer.py`) uses a Concrete Syntax Tree (`libcst`) to perform static analysis on each code block, determining what variables/functions it `provides` and what it `requires`.
    - This information is fed into a `BlockGraph` (`block_graph.py`), which builds a dependency graph of the blocks _within that single file_.

5.  **Incremental Execution**:

    - The `BlockGraph` identifies the set of "dirty" blocks—the block that was directly changed plus any blocks that depend on it.
    - The `IncrementalExecutor` (`incremental_executor.py`) is tasked with executing the document. It will only re-run the dirty blocks.
    - For any block that is _not_ dirty, it retrieves the previous result from the `BlockCache`.

6.  **Content-Addressable Caching**: The `BlockCache` (`block_cache.py`) uses a powerful caching strategy. The cache key for a block's result is a hash of its own content _plus the cache keys of all its dependencies_. This ensures that if a block or any code it depends on changes, the cache is automatically invalidated.

7.  **Streaming Results**: As each block is executed (or retrieved from cache), the `JsonDocumentGenerator` (`json_api.py`) converts the result into a JSON payload. This payload is immediately sent to the client via a `block-result` WebSocket message. This allows the UI to update progressively as the document executes.

8.  **Client Notification (Run End)**: Once all blocks in the file have been processed, the server sends a `run-end` message, signaling that the execution cycle is complete.

## Component Breakdown

The architecture is decoupled into several components, each with a clear responsibility:

- **`server.py`**: The main conductor. Contains the `LiveServer` class that initializes the HTTP and WebSocket servers, manages file watching, and orchestrates the overall workflow. It also contains the `ApiMiddleware` for serving document JSON to the frontend.

- **`client_registry.py`**: A simple but vital registry that tracks which WebSocket clients are currently watching which files. This ensures that updates are only sent to relevant clients.

- **`dependency_analyzer.py`**: The "code intelligence" layer. It uses `libcst` to parse Python code and understand the symbols each block provides and requires.

- **`block_graph.py`**: Manages **intra-file** dependencies (between blocks in one file). It is responsible for the topological sort of blocks for execution and for identifying the "dirty" downstream blocks after a change.

- **`file_graph.py`**: Manages **inter-file** dependencies (between `.py` files). It parses `import` statements to build a graph of how files relate to each other, defining the "blast radius" of any change.

- **`incremental_executor.py`**: The smart execution engine. It uses the `BlockGraph` and `BlockCache` to execute a document as efficiently as possible, skipping any work that has already been done and is still valid.

- **`block_cache.py`**: An intelligent, in-memory cache for block execution results. It is file-aware and can evict entries for files that are no longer being watched by any client to conserve memory.

- **`json_api.py`**: The presentation layer for the backend. These components are responsible for converting parsed documents, execution results, and file structures into the JSON format that the frontend SPA consumes.

## Design Philosophy

The complexity of this architecture is a direct result of its primary goal: **performance**. A naive implementation would re-execute the entire document on every change. This system goes to great lengths to minimize work, providing a near-instant feedback loop. The key principles are:

- **Decoupling**: Each component has a single responsibility and is unaware of the others' internal workings.
- **Accuracy**: Using a proper CST parser for dependency analysis is more robust than regex or other heuristics.
- **Efficiency**: The multi-layered caching and dependency analysis ensure that the minimum amount of code is re-run on any given change.
