import * as vscode from "vscode";
import { parser } from "@lezer/python";
import {
  SyntaxNode,
  TreeFragment,
  Input,
  Tree,
} from "@lezer/common";
import * as fs from "fs";

let config = vscode.workspace.getConfiguration("colight");

const defaultCellMarkers = ["# +", "# %%"];
const defaultBottomMarkers = ["# -", "# %-"];

function normalizeMarkers(markers: unknown, fallback: string[]): string[] {
  if (!Array.isArray(markers)) {
    return fallback.slice();
  }
  return markers.filter(
    (marker): marker is string =>
      typeof marker === "string" && marker.trim().length > 0
  );
}

function getEnabledCellMarkers(): string[] {
  const currentConfig = vscode.workspace.getConfiguration("colight");
  return normalizeMarkers(
    currentConfig.get("enabledCellMarkers"),
    defaultCellMarkers
  );
}

function lineStartsWithMarker(line: string, markers: readonly string[]): boolean {
  const trimmed = line.trimStart();
  return markers.some((marker) => trimmed.startsWith(marker));
}

function lineIsMarker(line: string, markers: readonly string[]): boolean {
  const trimmed = line.trim();
  return markers.some((marker) => trimmed === marker);
}

function isTopMarkerLine(line: string, markers?: readonly string[]): boolean {
  const activeMarkers = markers ?? getEnabledCellMarkers();
  return lineStartsWithMarker(line, activeMarkers);
}

function isBottomMarkerLine(line: string): boolean {
  return lineIsMarker(line, defaultBottomMarkers);
}

const logger = {
  time: (label: string) => {
    if (config.get("debugMode", false)) {
      console.time(label);
    }
  },
  timeEnd: (label: string) => {
    if (config.get("debugMode", false)) {
      console.timeEnd(label);
    }
  },
  log: (...args: any[]) => {
    if (config.get("debugMode", false)) {
      console.log(...args);
    }
  },
  error: (...args: any[]) => {
    if (config.get("debugMode", false)) {
      console.error(...args);
    }
  },
};

interface Cell {
  startLine: number;
  endLine: number;
  metadata: Record<string, string>;
  text: string;
}

function checkForExplicitMarkers(document: vscode.TextDocument): boolean {
  const markers = getEnabledCellMarkers();
  // Check only the first 100 lines for performance
  const linesToCheck = Math.min(document.lineCount, 100);
  for (let i = 0; i < linesToCheck; i++) {
    if (lineStartsWithMarker(document.lineAt(i).text, markers)) {
      return true;
    }
  }
  return false;
}

export function getCellAtPosition(
  document: vscode.TextDocument,
  position: vscode.Position,
  mode: "implicit" | "explicit" | "auto" = "auto"
): Cell | null {
  if (mode === "implicit") {
    return getImplicitCell(document, position);
  }
  if (mode === "explicit") {
    return checkForExplicitMarkers(document)
      ? getExplicitCell(document, position)
      : null;
  }
  // auto mode
  const useExplicitCells =
    config.get("useExplicitCellsIfPresent", false) &&
    checkForExplicitMarkers(document);
  return useExplicitCells
    ? getExplicitCell(document, position)
    : getImplicitCell(document, position);
}

function getExplicitCell(
  document: vscode.TextDocument,
  position: vscode.Position
): Cell | null {
  let startLine = position.line;
  let endLine = position.line;
  let metadata: Record<string, string> = {};
  const markers = getEnabledCellMarkers();

  const currentLine = document.lineAt(position.line).text;
  if (isTopMarkerLine(currentLine, markers)) {
    metadata = parseMetadata(currentLine);
    startLine++;
  } else {
    // Find the start of the cell
    while (startLine > 0) {
      const line = document.lineAt(startLine - 1).text;
      if (isTopMarkerLine(line, markers)) {
        metadata = parseMetadata(line);
        break;
      }
      startLine--;
    }
  }

  // Find the end of the cell
  while (endLine < document.lineCount - 1) {
    const line = document.lineAt(endLine + 1).text;
    if (isTopMarkerLine(line, markers) || isBottomMarkerLine(line)) {
      break;
    }
    endLine++;
  }

  let text = document.getText(
    new vscode.Range(
      startLine,
      0,
      endLine,
      document.lineAt(endLine).text.length
    )
  );
  return { startLine, endLine, metadata, text };
}

class StringInput implements Input {
  constructor(private content: string) {}

  get length() {
    return this.content.length;
  }

  chunk(from: number) {
    return this.content.slice(from);
  }

  read(from: number, to: number) {
    return this.content.slice(from, to);
  }

  lineChunks = false;
}

interface ParseTreeState {
  tree: Tree;
  version: number;
}

interface DocumentState {
  parseTree?: ParseTreeState;
  selectionStack: vscode.Selection[];
  activeFlashes: Set<vscode.Range>;
  hasExplicitCells: boolean | null; // null means not yet checked
}

const documentStates = new Map<string, DocumentState>();

function getDocumentState(
  document: vscode.TextDocument | string
): DocumentState {
  const key = typeof document === "string" ? document : document.uri.toString();
  if (!documentStates.has(key)) {
    documentStates.set(key, {
      selectionStack: [],
      activeFlashes: new Set(),
      hasExplicitCells: null,
    });
  }
  return documentStates.get(key)!;
}

function hasExplicitCells(document: vscode.TextDocument): boolean {
  const state = getDocumentState(document);
  if (state.hasExplicitCells === null) {
    state.hasExplicitCells = checkForExplicitMarkers(document);
  }
  return state.hasExplicitCells;
}

function getParseTree(
  document: vscode.TextDocument,
  changes?: readonly vscode.TextDocumentContentChangeEvent[]
): Tree {
  const state = getDocumentState(document);

  // Return cached tree if up to date
  if (state.parseTree && state.parseTree.version === document.version) {
    return state.parseTree.tree;
  }

  if (changes && state.parseTree) {
    // Incremental parsing
    logger.time("Incremental parsing");
    const mappedChanges = changes.map((change) => ({
      fromA: document.offsetAt(change.range.start),
      toA: document.offsetAt(change.range.end),
      fromB: document.offsetAt(change.range.start),
      toB: document.offsetAt(change.range.start) + change.text.length,
    }));

    const fragments = TreeFragment.applyChanges(
      TreeFragment.addTree(state.parseTree.tree),
      mappedChanges
    );
    state.parseTree = {
      tree: parser.parse(new StringInput(document.getText()), fragments),
      version: document.version,
    };
    logger.timeEnd("Incremental parsing");
    return state.parseTree.tree;
  } else {
    // Full parsing
    logger.time("Full parsing");
    state.parseTree = {
      tree: parser.parse(new StringInput(document.getText())),
      version: document.version,
    };
    logger.timeEnd("Full parsing");
    return state.parseTree.tree;
  }
}

function getImplicitCell(
  document: vscode.TextDocument,
  position: vscode.Position
): Cell | null {
  if (document.languageId !== "python") {return null;};

  const tree = getParseTree(document);
  let cursor = tree.cursor();

  // Move cursor to the position
  cursor.moveTo(document.offsetAt(position));

  // Find the top-level node at or before the cursor position
  while (cursor.parent()) {
    if (cursor.node.parent === null) {break;};
  }

  // If we're at the Script node, find the first child that starts after the position
  if (cursor.type.name === "Script") {
    cursor.firstChild();
    while (
      cursor.from <= document.offsetAt(position) &&
      cursor.nextSibling()
    ) {}
    if (cursor.from > document.offsetAt(position) && cursor.prevSibling()) {
    }
  }

  if (!cursor.node) {return null;};

  let startNode = cursor.node;
  let endNode = cursor.node;
  let onlyCommentNodes = isCommentNode(startNode);
  const code = document.getText();

  // Traverse backwards to find the start of the cell
  while (
    startNode.prevSibling &&
    !boundaryBetween(code, startNode, startNode.prevSibling)
  ) {
    startNode = startNode.prevSibling;
    onlyCommentNodes = onlyCommentNodes && isCommentNode(startNode);
  }

  // Traverse forwards to find the end of the cell
  while (
    endNode.nextSibling &&
    !boundaryBetween(code, endNode, endNode.nextSibling)
  ) {
    endNode = endNode.nextSibling;
    onlyCommentNodes = onlyCommentNodes && isCommentNode(endNode);
  }

  const startPos = document.positionAt(startNode.from);
  const endPos = document.positionAt(endNode.to);
  const startLine = startPos.line;
  let endLine = endPos.line;

  let text = document.getText(
    new vscode.Range(
      startLine,
      0,
      endLine,
      document.lineAt(endLine).text.length
    )
  );

  // Remove trailing newlines from cell
  const trailingNewlines = text.match(/\n*$/)?.[0].length ?? 0;
  text = text.replace(/\n+$/, "");
  endLine -= trailingNewlines;

  return {
    startLine,
    endLine,
    metadata: {},
    text,
  };
}

function isCommentNode(node: SyntaxNode): boolean {
  return node.type.name === "Comment";
}

function boundaryBetween(
  code: string,
  fromNode: SyntaxNode,
  toNode: SyntaxNode
): boolean {
  if (!fromNode || !toNode) {
    return false;
  }

  // First check if we're looking at a cell marker ("# %%")
  const isGoingUp = fromNode.from > toNode.from;
  const targetNode = isGoingUp ? toNode : toNode;
  const targetText = code.slice(targetNode.from, targetNode.to);
  const markers = getEnabledCellMarkers();
  if (lineStartsWithMarker(targetText, markers)) {
    return true;
  }

  // Then check for double newlines between nodes
  // Order nodes so we always process them start->end
  const [startNode, endNode] = isGoingUp 
    ? [toNode, fromNode] 
    : [fromNode, toNode];

  const textFrom = code.slice(startNode.from, startNode.to);
  const textBetween = code.slice(startNode.to, endNode.from);

  const firstNodeNewline = textFrom.endsWith("\n") ? 1 : 0;
  const totalNewlines = firstNodeNewline + (textBetween.match(/\n/g) || []).length;

  return totalNewlines > 1;
}

export function parseMetadata(line: string): Record<string, string> {
  const metadata: Record<string, string> = {};

  // First check for [markdown] or [md] format
  const typeMatch = line.match(/\[([a-zA-Z\-]+)\]/i);
  if (typeMatch) {
    metadata.type = typeMatch[1];
  }

  // Then check for key=value pairs
  const matches = line.match(/(\w+)\s*=\s*"([^"]*)"/g);
  if (matches) {
    matches.forEach((match) => {
      const [key, value] = match
        .split("=")
        .map((s) => s.trim().replace(/"/g, ""));
      metadata[key] = value;
    });
  }
  return metadata;
}

const decorations = {
  createDecoration: vscode.window.createTextEditorDecorationType,
  types: {} as Record<string, vscode.TextEditorDecorationType>,

  initTypes() {
    const accentColor = new vscode.ThemeColor("colight.accentColor");
    const cellBorderWidth = config.get("currentCell.borderWidth", "0");

    this.types = {
      cellTopAbove: this.createDecoration({
        isWholeLine: true,
        borderColor: accentColor,
        borderStyle: "solid",
        borderWidth: `0 0 ${cellBorderWidth} 0`,
      }),
      cellTopOn: this.createDecoration({
        isWholeLine: true,
        borderColor: accentColor,
        borderStyle: "solid",
        borderWidth: `${cellBorderWidth} 0 0 0`,
      }),
      cellBottomBelow: this.createDecoration({
        borderColor: accentColor,
        borderStyle: "solid",
        borderWidth: `${cellBorderWidth} 0 0 0`,
        isWholeLine: true,
      }),
      cellBottomOn: this.createDecoration({
        borderColor: accentColor,
        borderStyle: "solid",
        borderWidth: `0 0 ${cellBorderWidth} 0`,
        isWholeLine: true,
      }),
      evaluating: this.createDecoration({
        backgroundColor: accentColor,
        isWholeLine: true,
        overviewRulerColor: accentColor,
        overviewRulerLane: vscode.OverviewRulerLane.Full,
      }),
    };
  },

  apply(
    editor: vscode.TextEditor,
    range: vscode.Range | null,
    decorationType: vscode.TextEditorDecorationType
  ) {
    if (range) {
      const state = getDocumentState(editor.document);
      if (decorationType === this.types.evaluating) {
        state.activeFlashes.add(range);
      }
      editor.setDecorations(
        decorationType,
        decorationType === this.types.evaluating
          ? Array.from(state.activeFlashes)
          : [range]
      );
    }
  },

  getCellRange(
    editor: vscode.TextEditor,
    cell: Cell | null
  ): vscode.Range | null {
    if (!cell) {return null;};
    return new vscode.Range(
      cell.startLine,
      0,
      cell.endLine,
      editor.document.lineAt(cell.endLine).text.length
    );
  },

  decorateCurrentCell(editor: vscode.TextEditor, cell: Cell | null) {
    this.clearAllDecorations(editor);

    if (!config.get("currentCell.show", true)) {
      return;
    }

    const range = this.getCellRange(editor, cell);
    if (range) {
      const isFirstLine = range.start.line === 0;
      const isLastLine = range.end.line === editor.document.lineCount - 1;

      // Top border
      if (isFirstLine) {
        this.apply(
          editor,
          new vscode.Range(range.start, range.start),
          this.types.cellTopOn
        );
      } else {
        const topLineRange = new vscode.Range(
          range.start.line - 1,
          editor.document.lineAt(range.start.line - 1).text.length,
          range.start.line - 1,
          editor.document.lineAt(range.start.line - 1).text.length
        );
        this.apply(editor, topLineRange, this.types.cellTopAbove);
      }

      // Bottom border
      if (isLastLine) {
        this.apply(
          editor,
          new vscode.Range(range.end, range.end),
          this.types.cellBottomOn
        );
      } else {
        const bottomLineRange = new vscode.Range(
          range.end.line + 1,
          0,
          range.end.line + 1,
          0
        );
        this.apply(editor, bottomLineRange, this.types.cellBottomBelow);
      }
    }
  },

  flashCell(
    editor: vscode.TextEditor,
    cell: Cell | null
  ): Promise<vscode.Disposable> {
    return new Promise((resolve) => {
      const range = this.getCellRange(editor, cell);
      if (range) {
        this.apply(editor, range, this.types.evaluating);
        const state = getDocumentState(editor.document);
        const disposable = {
          dispose: () => {
            state.activeFlashes.delete(range);
            editor.setDecorations(
              this.types.evaluating,
              Array.from(state.activeFlashes)
            );
          },
        };
        setTimeout(() => resolve(disposable), 200);
      } else {
        resolve({ dispose: () => {} });
      }
    });
  },

  clearAllDecorations(editor: vscode.TextEditor) {
    const state = getDocumentState(editor.document);
    state.activeFlashes.clear();
    Object.values(this.types).forEach((decorationType) => {
      editor.setDecorations(decorationType, []);
    });
  },

  dispose() {
    Object.values(this.types).forEach((decorationType) =>
      decorationType.dispose()
    );
  },
};

function renderCommentsAsMarkdown(
  document: vscode.TextDocument,
  cell: Cell
): string {
  const import_markdown = "# \nfrom IPython.display import display, Markdown";

  function wrapMarkdown(text: string) {
    if (/^\s*-[-*]-/.test(text)) {
      return text
        .split("\n")
        .map((line) => `# -${line}`)
        .join("\n");
    } else {
      return `# \ndisplay(Markdown(${JSON.stringify(text)}))`;
    }
  }

  const tree = getParseTree(document);
  const cellStartOffset = document.offsetAt(
    new vscode.Position(cell.startLine, 0)
  );
  const cellEndOffset = document.offsetAt(
    new vscode.Position(cell.endLine, document.lineAt(cell.endLine).text.length)
  );

  let processedLines: string[] = [];
  let markdownChunk: string[] = [];
  let hasMarkdown = false;

  function processMarkdownChunk() {
    const markdownContent = markdownChunk
      .filter((line) => line.trim() !== "")
      .join("\n");
    if (markdownContent) {
      processedLines.push(wrapMarkdown(markdownContent));
      hasMarkdown = true;
    }
    markdownChunk = [];
  }

  let cursor = tree.cursor();
  cursor.moveTo(cellStartOffset);

  // Move up to Script level if we're not already there
  while (cursor.node.parent !== null) {
    cursor.parent();
  }

  // Assert we're at the Script level
  if (cursor.type.name !== "Script") {
    throw new Error("Cell parsing error: Not at top level Script node");
  }

  // Move to first child that's within our cell
  if (!cursor.firstChild()) {return processedLines.join("\n");};

  // Create a new cursor for iterating through children
  let childCursor = cursor;
  while (childCursor.from < cellStartOffset && childCursor.nextSibling()) {}

  // Process all siblings within the cell bounds
  do {
    if (childCursor.from >= cellEndOffset) {break;};

    if (childCursor.type.name === "Comment") {
      // Get the position of the comment
      const commentPos = document.positionAt(childCursor.from);
      // Only process comments that start at column 0 (beginning of line)
      if (commentPos.character === 0) {
        const commentText = document.getText(
          new vscode.Range(
            document.positionAt(childCursor.from),
            document.positionAt(childCursor.to)
          )
        );
        markdownChunk.push(commentText.slice(1).trim());
      } else {
        // For non-starting comments, treat them as regular code
        const commentText = document.getText(
          new vscode.Range(
            document.positionAt(childCursor.from),
            document.positionAt(childCursor.to)
          )
        );
        processedLines.push(commentText);
      }
    } else {
      if (markdownChunk.length > 0) {
        processMarkdownChunk();
      }
      const nodeText = document.getText(
        new vscode.Range(
          document.positionAt(childCursor.from),
          document.positionAt(childCursor.to)
        )
      );
      processedLines.push(nodeText);
    }
  } while (childCursor.nextSibling());

  if (markdownChunk.length > 0) {
    processMarkdownChunk();
  }

  if (hasMarkdown) {
    processedLines.unshift(import_markdown);
  }

  return processedLines.join("\n");
}

function stripComments(cellText: string): string {
  return cellText
    .split("\n")
    .filter((line) => !line.startsWith("#"))
    .join("\n");
}

function stripMarkdownCommentMarkers(cellText: string): string {
  return cellText
    .split("\n")
    .map((line) => line.replace(/^#\s*/, ""))
    .join("\n");
}

async function evaluateCell(
  editor: vscode.TextEditor,
  cell: Cell | null,
  renderComments?: boolean
): Promise<void> {
  if (!cell) {
    vscode.window.showErrorMessage("No cell found at cursor position");
    return;
  }

  const isExplicitCell = hasExplicitCells(editor.document);
  let source: string;

  if (isExplicitCell) {
    // For explicit cells, check metadata for markdown
    if (cell.metadata.type === "markdown") {
      // Strip just the markdown comment markers
      const markdownText = stripMarkdownCommentMarkers(cell.text);
      source = `from IPython.display import display, Markdown\ndisplay(Markdown(${JSON.stringify(
        markdownText
      )}))`;
    } else {
      source = stripComments(cell.text);
    }
  } else {
    // For implicit cells, use the existing comment rendering logic
    source =
      renderComments ?? config.get("renderComments", true)
        ? renderCommentsAsMarkdown(editor.document, cell)
        : stripComments(cell.text);
  }

  // Return early if the source is blank
  if (!source.trim()) {
    return;
  }

  const flashDisposable = decorations.flashCell(editor, cell);

  try {
    await vscode.commands.executeCommand(
      "jupyter.execSelectionInteractive",
      source
    );
  } finally {
    (await flashDisposable).dispose();
  }
}

function moveToNextExplicitCell(
  editor: vscode.TextEditor,
  currentCell: Cell
): void {
  const document = editor.document;
  let nextLine = currentCell.endLine + 1;
  const markers = getEnabledCellMarkers();

  while (nextLine < document.lineCount) {
    const line = document.lineAt(nextLine).text;
    if (isTopMarkerLine(line, markers)) {
      const newPosition = new vscode.Position(nextLine, 0);
      editor.selection = new vscode.Selection(newPosition, newPosition);
      editor.revealRange(
        new vscode.Range(newPosition, newPosition),
        vscode.TextEditorRevealType.InCenterIfOutsideViewport
      );
      break;
    }
    nextLine++;
  }
}

function moveToNextCell(editor: vscode.TextEditor, currentCell: Cell): void {
  if (!currentCell) {
    return;
  }
  if (
    config.get("useExplicitCellsIfPresent", false) &&
    checkForExplicitMarkers(editor.document)
  ) {
    moveToNextExplicitCell(editor, currentCell);
    return;
  }
  const document = editor.document;
  let nextLine = currentCell.endLine + 1;

  while (
    nextLine < document.lineCount &&
    document.lineAt(nextLine).isEmptyOrWhitespace
  ) {
    nextLine++;
  }

  if (nextLine < document.lineCount) {
    const nextCell = getCellAtPosition(
      document,
      new vscode.Position(nextLine, 0)
    );
    if (nextCell) {
      const newPosition = new vscode.Position(nextCell.startLine, 0);
      editor.selection = new vscode.Selection(newPosition, newPosition);

      // Calculate the range to reveal, including the next cell and some context
      const endOfNextCell = new vscode.Position(
        nextCell.endLine,
        document.lineAt(nextCell.endLine).text.length
      );
      const rangeToReveal = new vscode.Range(newPosition, endOfNextCell);

      // Reveal the range with a small amount of padding above and below
      editor.revealRange(
        rangeToReveal,
        vscode.TextEditorRevealType.InCenterIfOutsideViewport
      );
    }
  }
}

function parseCells(document: vscode.TextDocument, upToLine?: number): Cell[] {
  const cells: Cell[] = [];
  let currentLine = 0;
  const lastLine =
    upToLine !== undefined
      ? Math.min(upToLine, document.lineCount - 1)
      : document.lineCount - 1;

  while (currentLine <= lastLine) {
    const cell = getCellAtPosition(
      document,
      new vscode.Position(currentLine, 0)
    );
    if (cell) {
      cells.push(cell);
      // Move to the next non-empty line after the current cell
      currentLine = cell.endLine + 1;
      while (
        currentLine <= lastLine &&
        document.lineAt(currentLine).isEmptyOrWhitespace
      ) {
        currentLine++;
      }
    } else {
      // If no cell is found, move to the next non-empty line
      do {
        currentLine++;
      } while (
        currentLine <= lastLine &&
        document.lineAt(currentLine).isEmptyOrWhitespace
      );
    }
  }

  return cells;
}

async function evaluateAllCells(editor: vscode.TextEditor): Promise<void> {
  for (const cell of parseCells(editor.document)) {
    await evaluateCell(editor, cell, false);
  }
}

async function evaluateCellsAboveAndCurrent(
  editor: vscode.TextEditor
): Promise<void> {
  // TODO
  // figure out how to run cells in the active kernel without showing the result
  // in the interactive window. Then we should be able to show the currently active
  // cell in the scrollbar area.

  const document = editor.document;
  const currentPosition = editor.selection.active;

  const currentCell = getCellAtPosition(document, currentPosition);
  if (!currentCell) {
    return; // Exit if no current cell is found
  }

  const originalSelection = editor.selection;
  // jupyter.runtoline runs up to the previous line,
  // so we must set our selection to the next line.
  const runUntilLine = Math.min(
    currentCell.endLine + 1,
    document.lineCount
  );
  const safeRunUntilPosition = document.validatePosition(
    new vscode.Position(runUntilLine, 0)
  );
  editor.selection = new vscode.Selection(
    safeRunUntilPosition,
    safeRunUntilPosition
  );
  await vscode.commands.executeCommand("jupyter.runtoline");
  editor.selection = originalSelection;
}

function isStandardEditor(editor: vscode.TextEditor | undefined): boolean {
  return (
    editor !== undefined &&
    editor.document.languageId === "python" &&
    editor.viewColumn !== undefined
  );
}

async function evaluateSelection(editor: vscode.TextEditor): Promise<void> {
  if (!editor.selection.isEmpty) {
    // If there's a selection, evaluate it
    const selectedText = editor.document.getText(editor.selection);
    await evaluateCell(editor, {
      startLine: editor.selection.start.line,
      endLine: editor.selection.end.line,
      metadata: {},
      text: selectedText,
    });
  }
}

function exportNotebook(document: vscode.TextDocument): any {
  const cells = parseCells(document);
  const notebookCells = cells.map((cell) => ({
    source: cell.text.split("\n"),
    metadata: cell.metadata,
    outputs: [],
  }));

  return {
    nbformat: 4,
    nbformat_minor: 2,
    metadata: {
      kernelspec: {
        display_name: "Python 3",
        language: "python",
        name: "python3",
      },
      language_info: {
        codemirror_mode: {
          name: "ipython",
          version: 3,
        },
        file_extension: ".py",
        mimetype: "text/x-python",
        name: "python",
        nbconvert_exporter: "python",
        pygments_lexer: "ipython3",
        version: "3.8.0",
      },
    },
    cells: notebookCells,
  };
}

function nodeAtCursor(cursorOffset: number, tree: Tree): SyntaxNode | null {
  let cursor = tree.cursor();
  let currentNode = cursor.moveTo(cursorOffset).node;
  let prevNode = cursor.moveTo(cursorOffset, -1).node;
  let nextNode = cursor.moveTo(cursorOffset, 1).node;

  return [prevNode, nextNode, currentNode]
    .filter((node): node is SyntaxNode => node !== null)
    .reduce((smallest, node) =>
      !smallest || node.to - node.from < smallest.to - smallest.from
        ? node
        : smallest
    );
}

function getCellRange(
  document: vscode.TextDocument,
  position: vscode.Position,
  mode: "implicit" | "explicit" | "auto" = "auto"
): vscode.Range | null {
  const cell = getCellAtPosition(document, position, mode);
  if (!cell) {
    return null;
  }
  return new vscode.Range(
    new vscode.Position(cell.startLine, 0),
    new vscode.Position(
      cell.endLine,
      document.lineAt(cell.endLine).text.trimEnd().length
    )
  );
}
function isLargerThan(range: vscode.Range, other: vscode.Range): boolean {
  return (
    range.end.line - range.start.line > other.end.line - other.start.line ||
    (range.end.line - range.start.line === other.end.line - other.start.line &&
      range.end.character - range.start.character >
        other.end.character - other.start.character)
  );
}

/**
 * Gets the start and end offsets for a syntax node, adjusting for Python body nodes
 * by trimming colons, whitespace and trailing newlines.
 */
function nodeBounds(
  node: SyntaxNode,
  document: vscode.TextDocument
): { start: number; end: number } {
  let start = node.from;
  let end = node.to;

  // Get text content of node
  let nodeText = document.getText(
    new vscode.Range(document.positionAt(start), document.positionAt(end))
  );

  if (node.type.name === "Body") {
    // Trim : and whitespace from start of body, and newlines from end
    const startMatch = nodeText.match(/^:\s*/);
    if (startMatch) {
      start += startMatch[0].length;
    }
    const endMatch = nodeText.match(/\n+$/);
    if (endMatch) {
      end -= endMatch[0].length;
    }
  } else {
    // For other nodes, trim whitespace from end
    const endMatch = nodeText.match(/\s+$/);
    if (endMatch) {
      end -= endMatch[0].length;
    }
  }

  return { start, end };
}

function expandSelection(editor: vscode.TextEditor): void {
  if (editor.document.languageId !== "python") {return;};

  const document = editor.document;
  const currentSelection = editor.selection;
  const state = getDocumentState(document);

  state.selectionStack.push(currentSelection);
  const currentRange = new vscode.Range(
    currentSelection.start,
    currentSelection.end
  );

  // Get next syntax-based selection
  const tree = getParseTree(document);
  let syntaxNode: SyntaxNode | null = null;

  if (currentSelection.isEmpty) {
    const cursorOffset = document.offsetAt(currentSelection.active);
    syntaxNode = nodeAtCursor(cursorOffset, tree);
  } else {
    const selectionStart = document.offsetAt(currentSelection.start);
    const selectionEnd = document.offsetAt(currentSelection.end);

    let cursor = tree.cursor();
    cursor.moveTo(selectionStart, 1);

    while (cursor.node) {
      const bounds = nodeBounds(cursor.node, document);

      if (bounds.start < selectionStart || bounds.end > selectionEnd) {
        syntaxNode = cursor.node;
        break;
      }

      if (!cursor.parent()) {break;};
    }

    if (!syntaxNode) {
      syntaxNode = cursor.node;
    }
  }

  // Get all possible ranges
  const ranges = [
    // Syntax-based range
    syntaxNode &&
      new vscode.Range(
        document.positionAt(nodeBounds(syntaxNode, document).start),
        document.positionAt(nodeBounds(syntaxNode, document).end)
      ),
    // Cell-based ranges
    getCellRange(document, currentSelection.active, "implicit"),
    getCellRange(document, currentSelection.active, "explicit"),
  ]
    .filter((range): range is vscode.Range => range !== null)
    .sort((a, b) => {
      const aSize = a.end.line - a.start.line;
      const bSize = b.end.line - b.start.line;
      return aSize - bSize;
    });

  // Find index of first range larger than current
  const nextRangeIndex = ranges.findIndex((range) =>
    isLargerThan(range, currentRange)
  );

  if (nextRangeIndex >= 0) {
    const selectedRange = ranges[nextRangeIndex];
    editor.selection = new vscode.Selection(
      selectedRange.start,
      selectedRange.end
    );
  }
}

function shrinkSelection(editor: vscode.TextEditor): void {
  if (editor.document.languageId !== "python") {return;};

  const state = getDocumentState(editor.document);
  if (state.selectionStack.length > 0) {
    editor.selection = state.selectionStack.pop()!;
    return;
  }

  const document = editor.document;
  const selection = editor.selection;

  const tree = getParseTree(document);
  const cursor = tree.cursor();
  const selectionStart = document.offsetAt(selection.start);
  const selectionEnd = document.offsetAt(selection.end);

  cursor.moveTo(selectionStart, 1);

  let targetNode: SyntaxNode | null = null;
  while (cursor.node) {
    if (cursor.from >= selectionStart && cursor.to <= selectionEnd) {
      targetNode = cursor.node;
      if (
        !cursor.firstChild() ||
        cursor.from < selectionStart ||
        cursor.to > selectionEnd
      ) {
        break;
      }
    } else {
      if (!cursor.nextSibling()) {break;};
    }
  }

  if (
    targetNode &&
    (targetNode.from !== selectionStart || targetNode.to !== selectionEnd)
  ) {
    const newSelection = new vscode.Selection(
      document.positionAt(targetNode.from),
      document.positionAt(targetNode.to)
    );
    editor.selection = newSelection;
  }
}

function clearSelectionStack(editor: vscode.TextEditor) {
  const state = getDocumentState(editor.document);
  state.selectionStack = [];
}

export function activate(context: vscode.ExtensionContext) {
  try {
    let disposables: vscode.Disposable[] = [];

    decorations.initTypes();

    // Initial parse
    const activeEditor = vscode.window.activeTextEditor;
    if (activeEditor && isStandardEditor(activeEditor)) {
      getParseTree(activeEditor.document);
    }

    disposables.push(
      vscode.commands.registerCommand("extension.evaluateCell", async () => {
        const editor = vscode.window.activeTextEditor;
        if (editor && isStandardEditor(editor)) {
          if (!editor.selection.isEmpty) {
            await evaluateSelection(editor);
          } else {
            await evaluateCell(
              editor,
              getCellAtPosition(editor.document, editor.selection.active)
            );
          }
        }
      })
    );

    disposables.push(
      vscode.commands.registerCommand(
        "extension.evaluateImplicitCell",
        async () => {
          const editor = vscode.window.activeTextEditor;
          if (editor && isStandardEditor(editor)) {
            if (!editor.selection.isEmpty) {
              await evaluateSelection(editor);
            } else {
              await evaluateCell(
                editor,
                getImplicitCell(editor.document, editor.selection.active)
              );
            }
          }
        }
      )
    );

    disposables.push(
      vscode.commands.registerCommand(
        "extension.evaluateCellAndMoveNext",
        async () => {
          const editor = vscode.window.activeTextEditor;
          if (editor && isStandardEditor(editor)) {
            const cell = getCellAtPosition(
              editor.document,
              editor.selection.active
            );

            if (cell) {
              await evaluateCell(editor, cell);
              moveToNextCell(editor, cell);
            }
          }
        }
      )
    );

    disposables.push(
      vscode.commands.registerCommand(
        "extension.evaluateCellAndAbove",
        async () => {
          const editor = vscode.window.activeTextEditor;
          if (editor && isStandardEditor(editor)) {
            await evaluateCellsAboveAndCurrent(editor);
          }
        }
      )
    );

    disposables.push(
      vscode.workspace.onDidChangeTextDocument((event) => {
        const state = getDocumentState(event.document);
        state.hasExplicitCells = null; // Reset the check when document changes
        getParseTree(event.document, event.contentChanges);
      })
    );

    disposables.push(
      vscode.window.onDidChangeTextEditorSelection((event) => {
        const editor = event.textEditor;
        if (isStandardEditor(editor)) {
          const cell = getCellAtPosition(
            editor.document,
            editor.selection.active
          );
          decorations.decorateCurrentCell(editor, cell);

          // Clear the selection stack if the selection change wasn't caused by our commands
          if (!event.kind) {
            clearSelectionStack(editor);
          }
        }
      })
    );

    disposables.push(
      vscode.workspace.onDidChangeConfiguration((e) => {
        if (e.affectsConfiguration("colight")) {
          config = vscode.workspace.getConfiguration("colight");
          for (const state of documentStates.values()) {
            state.hasExplicitCells = null;
          }
          decorations.dispose();
          decorations.initTypes();
        }
      })
    );

    disposables.push(
      vscode.commands.registerCommand("extension.exportNotebook", async () => {
        const editor = vscode.window.activeTextEditor;
        if (editor && isStandardEditor(editor)) {
          const notebook = exportNotebook(editor.document);
          const currentFilePath = editor.document.fileName;
          const notebookPath = currentFilePath.replace(/\.py$/, ".ipynb");

          fs.writeFile(
            notebookPath,
            JSON.stringify(notebook, null, 2),
            (err) => {
              if (err) {
                vscode.window.showErrorMessage(
                  `Failed to save notebook: ${err.message}`
                );
              } else {
                vscode.window.showInformationMessage(
                  `Notebook saved to ${notebookPath}`
                );
              }
            }
          );
        }
      })
    );

    disposables.push(
      vscode.commands.registerCommand("extension.expandSelection", () => {
        const editor = vscode.window.activeTextEditor;
        if (editor && isStandardEditor(editor)) {
          expandSelection(editor);
        }
      })
    );

    disposables.push(
      vscode.commands.registerCommand("extension.shrinkSelection", () => {
        const editor = vscode.window.activeTextEditor;
        if (editor && isStandardEditor(editor)) {
          shrinkSelection(editor);
        }
      })
    );

    disposables.push(
      vscode.workspace.onDidChangeTextDocument(() => {
        // Clear the selection stack when the document changes
        if (vscode.window.activeTextEditor) {
          clearSelectionStack(vscode.window.activeTextEditor);
        }
      })
    );

    disposables.push(
      vscode.commands.registerCommand(
        "extension.evaluateAllCells",
        async () => {
          const editor = vscode.window.activeTextEditor;
          if (editor && isStandardEditor(editor)) {
            await evaluateAllCells(editor);
          }
        }
      )
    );

    disposables.push(
      vscode.workspace.onDidCloseTextDocument((document) => {
        // Clean up all document-specific state
        documentStates.delete(document.uri.toString());
      })
    );

    context.subscriptions.push(...disposables);
  } catch (error) {
    logger.error("Error activating extension:", error);
    vscode.window.showErrorMessage("Failed to activate Colight extension");
  }
}

export function deactivate() {
  decorations.dispose();
}
