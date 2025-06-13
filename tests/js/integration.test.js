import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { JSDOM } from 'jsdom';
import { parseColightData } from '../../src/colight/js/format.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const testArtifactsDir = join(__dirname, '..', 'test-artifacts');
const testColightFile = join(testArtifactsDir, 'test-raster.colight');

// Mock global objects for testing
global.window = new JSDOM().window;
global.document = window.document;
global.HTMLElement = window.HTMLElement;

describe('Colight Integration', () => {
  beforeAll(() => {
    if (!existsSync(testColightFile)) {
      throw new Error(
        `Test artifact ${testColightFile} not found. Make sure Python tests run first.`
      );
    }
  });

  it('should parse .colight file and have renderable data structure', () => {
    const fileData = readFileSync(testColightFile);
    const data = parseColightData(fileData);
    const {buffers} = data;

    // Verify the data structure is what we expect for rendering
    expect(data.ast).toBeDefined();
    expect(data.initialState).toBeDefined();
    expect(data.syncedKeys).toBeDefined();
    expect(data.listeners).toBeDefined();
    expect(data.imports).toBeDefined();

    // Verify buffer system works
    expect(buffers.length).toBeGreaterThan(0);

    // Check that we can find buffer references in the AST
    const hasBufferRefs = JSON.stringify(data).includes('"__type__":"buffer"') ||
                         JSON.stringify(data).includes('"__buffer_index__"');
    expect(hasBufferRefs).toBe(true);
  });

  it('should maintain consistency between buffer layout and actual buffers', () => {
    const fileData = readFileSync(testColightFile);
    const data = parseColightData(fileData);
    const {buffers} = data;

    const layout = data.bufferLayout;

    // Verify layout consistency
    expect(layout.offsets.length).toBe(layout.count);
    expect(layout.lengths.length).toBe(layout.count);
    expect(buffers.length).toBe(layout.count);

    // Verify offsets are sequential
    let expectedOffset = 0;
    for (let i = 0; i < layout.count; i++) {
      expect(layout.offsets[i]).toBe(expectedOffset);
      expect(buffers[i].length).toBe(layout.lengths[i]);
      expectedOffset += layout.lengths[i];
    }

    expect(expectedOffset).toBe(layout.totalSize);
  });

  it('should have valid raster data structure for the test file', () => {
    const fileData = readFileSync(testColightFile);
    const data = parseColightData(fileData);
    const {buffers} = data;

    // Since this is a raster plot, we should find raster-related structure
    // Check both AST and initialState since raster data might be in state
    const fullDataStr = JSON.stringify(data);
    expect(fullDataStr).toMatch(/raster|image|ndarray|PlotSpec/i);

    // Should have at least one buffer for the raster data
    expect(buffers.length).toBeGreaterThanOrEqual(1);

    // Buffers should contain actual data
    const totalBytes = buffers.reduce((sum, buf) => sum + buf.length, 0);
    expect(totalBytes).toBeGreaterThan(0);

    // Should have buffer references in the data
    const hasBufferRefs = fullDataStr.includes('"__type__":"buffer"') ||
                         fullDataStr.includes('"__buffer_index__"') ||
                         fullDataStr.includes('"__type__":"ndarray"');
    expect(hasBufferRefs).toBe(true);
  });
});
