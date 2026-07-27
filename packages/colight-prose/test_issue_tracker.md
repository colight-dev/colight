# Colight-Site Test Issue Tracker

## Overview

This document tracks all identified issues with the test suite in the colight-prose package, categorizing them by priority and type.

## Critical Issues (High Priority)

### 1. Missing Core Functionality Tests

- **Issue**: The main document generation flow (`/api/document/<path>`) has NO tests
- **Impact**: Core functionality could break without detection
- **Files affected**: `colight_prose/json_api.py` (ApiMiddleware)
- **Action**: Create integration tests that:
  - Request document generation for Python files
  - Verify block execution and result streaming
  - Test error handling during execution

### 2. Live Server Integration Gaps

- **Issue**: No automated tests for the edit → save → execute → stream workflow
- **Impact**: Development workflow breakages go undetected
- **Files affected**: `colight_prose/server.py`, `colight_prose/incremental_executor.py`
- **Action**: Create end-to-end tests with real file watching and WebSocket streaming

### 3. API Endpoints Not Tested

- **Issue**: Critical API endpoints have no test coverage
- **Impact**: API changes could break client functionality
- **Endpoints missing tests**:
  - `/api/files` - File listing
  - `/api/index` - File tree structure
  - `/api/document/<path>` - Document generation
  - `/api/visual/<id>` - Visual data serving
- **Action**: Add API-level tests for each endpoint

## Structural Issues (Medium Priority)

### 4. Test Organization Problems

- **Issue**: Tests are not clearly organized by type or purpose
- **Current structure**: Flat directory with mixed unit/integration tests
- **Proposed structure**:
  ```
  tests/
  ├── unit/           # Pure unit tests (single module/function)
  │   ├── live/       # Tests for colight_prose modules
  │   ├── site/       # Tests for colight_prose modules
  │   └── static/     # Tests for colight_prose.static modules
  ├── integration/    # Tests that span multiple modules
  ├── e2e/           # End-to-end tests with real server
  ├── fixtures/      # Test data and helpers
  └── examples/      # Demo files (not actual tests)
  ```

### 5. Conceptual/Exploratory Tests

- **Issue**: Some tests don't test our code, but test library/language behavior
- **Files to review**:
  - `test_client_registration.py` - Implements its own ClientRegistry instead of testing ours
  - `test_hash_implementations.py` - Partially exploratory
  - `live-integration.test.js` - Contains JavaScript behavior tests
- **Action**: Move exploratory tests to documentation or remove

### 6. Example Files Misplaced

- **Issue**: `tests/examples/` contains mostly demo files, not test fixtures
- **Only used in tests**: `visual_update.py`, `visual_update_dep.py`
- **Action**: Move demos to `docs/examples/` or `demo/` directory

## Missing Test Coverage (Medium Priority)

### 7. Error Handling Gaps

- **Missing tests for**:
  - Python syntax errors in served files
  - Import errors and circular dependencies
  - File not found scenarios
  - Malformed WebSocket messages
  - Server crash recovery
- **Action**: Add negative test cases

### 8. WebSocket Protocol Testing

- **Missing tests for**:
  - Complete message flow (run-start → block-results → run-end)
  - Message cancellation
  - Client disconnection during execution
  - Multiple clients with different run versions
- **Action**: Create WebSocket protocol test suite

### 9. File System Operations

- **Missing tests for**:
  - File creation/deletion during watch
  - Symlink handling
  - Permission errors
  - Large file handling
- **Action**: Add file system edge case tests

### 10. Performance and Concurrency

- **Missing tests for**:
  - Multiple clients watching different files
  - Rapid file changes
  - Memory usage under load
  - Block cache eviction under pressure
- **Action**: Add performance regression tests

## Code Quality Issues (Low Priority)

### 11. JavaScript Test Gaps

- **CommandBar component**: Only basic rendering tested, missing:
  - Search functionality
  - File navigation
  - Recent files feature
  - Keyboard navigation
- **Action**: Expand component test coverage

### 12. Test Naming Inconsistency

- **Issue**: Mixed naming conventions
  - Python: `test_*.py`
  - JavaScript: Some use `*.test.js`, others don't
- **Action**: Standardize naming conventions

### 13. TypeScript Usage Inconsistent

- **Issue**: Some tests use `.jsx`, others `.tsx`
- **Action**: Standardize on TypeScript for all JavaScript tests

### 14. Debug Code in Tests

- **Issue**: Some tests have leftover console.log statements
- **Files**: `live-integration.test.js`
- **Action**: Clean up debug code

## Test Improvements

### 15. Coverage Tooling

- **Issue**: Coverage reports are difficult to generate
- **Action**: Configure pytest-cov properly in pyproject.toml

### 16. Test Documentation

- **Issue**: No README explaining test structure and conventions
- **Action**: Create tests/README.md with:
  - Test organization guide
  - How to run different test suites
  - Coverage requirements
  - Testing best practices

## Recommended Actions by Priority

### Immediate (Block Release)

1. Add tests for `/api/document/<path>` endpoint
2. Add basic live server integration test
3. Fix failing tests in `test_attribute_access_false_positives.py`

### Short Term (Next Sprint)

1. Reorganize test structure into unit/integration/e2e
2. Add API endpoint tests
3. Add WebSocket protocol tests
4. Move conceptual tests out of main test suite

### Medium Term (Next Quarter)

1. Add comprehensive error handling tests
2. Add performance regression tests
3. Expand JavaScript component tests
4. Add file system edge case tests

### Long Term (Ongoing)

1. Maintain test coverage above 80%
2. Add tests for all new features
3. Regular test suite cleanup and refactoring

## Testing Philosophy Recommendations

Based on the user's preference for "a smaller number of tests that cover involved pathways":

1. **Focus on Integration Tests**: Test complete user workflows rather than individual functions
2. **Test Critical Paths**: Prioritize tests for the edit → execute → view cycle
3. **Avoid Over-Mocking**: Use real components where possible
4. **Test Error Recovery**: Ensure the system fails gracefully
5. **Performance Regression Tests**: Catch performance degradations early

## Next Steps

1. Review this document and prioritize which issues to address
2. Create GitHub issues for high-priority items
3. Begin with adding tests for core untested functionality
4. Gradually reorganize test structure as new tests are added
