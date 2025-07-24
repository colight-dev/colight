// src/bylight.ts
interface BylightOptions {
  target?: string | HTMLElement;
  colorScheme?: string[];
}

interface HighlightOptions {
  matchId?: string;
  colorIndex?: number;
}

// Utility functions
function escapeRegExp(string: string): string {
  return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

const DefaultColors: string[] = [
  "#59a14f",
  "#b82efe",
  "#007bfe",
  "#6a6a6a",
  "#ff4245",
  "#7c2d00",
  "#76b7b2",
  "#d4af37",
  "#ff9da7",
  "#f28e2c",
];

function matchWildcard(
  text: string,
  startIndex: number,
  nextLiteral: string,
): number {
  let index = startIndex;
  let bracketDepth = 0;
  let inString: string | null = null;

  while (index < text.length) {
    if (inString) {
      if (text[index] === inString && text[index - 1] !== "\\") {
        inString = null;
      }
    } else if (text[index] === '"' || text[index] === "'") {
      inString = text[index];
    } else if (bracketDepth === 0 && text[index] === nextLiteral) {
      return index;
    } else if (
      text[index] === "(" ||
      text[index] === "[" ||
      text[index] === "{"
    ) {
      bracketDepth++;
    } else if (
      text[index] === ")" ||
      text[index] === "]" ||
      text[index] === "}"
    ) {
      if (bracketDepth === 0) {
        return index;
      }
      bracketDepth--;
    }
    index++;
  }
  return index;
}

function findMatches(text: string, pattern: string): [number, number][] {
  if (pattern.startsWith("/") && pattern.endsWith("/")) {
    return findRegexMatches(text, pattern.slice(1, -1));
  }

  const matches: [number, number][] = [];
  let currentPosition = 0;

  while (currentPosition < text.length) {
    const match = findSingleMatch(text, pattern, currentPosition);
    if (match) {
      matches.push(match);
      currentPosition = match[1];
    } else {
      currentPosition++;
    }
  }

  return matches;
}

function findRegexMatches(text: string, pattern: string): [number, number][] {
  const regex = new RegExp(pattern, "g");
  let matches: [number, number][] = [];
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    matches.push([match.index, regex.lastIndex]);
  }

  return matches;
}

function findSingleMatch(
  text: string,
  pattern: string,
  startPosition: number,
): [number, number] | null {
  let patternPosition = 0;
  let textPosition = startPosition;

  while (textPosition < text.length && patternPosition < pattern.length) {
    if (pattern.startsWith("...", patternPosition)) {
      const nextCharacter = pattern[patternPosition + 3] || "";
      textPosition = matchWildcard(text, textPosition, nextCharacter);
      patternPosition += 3;
    } else if (text[textPosition] === pattern[patternPosition]) {
      textPosition++;
      patternPosition++;
    } else {
      return null;
    }
  }

  return patternPosition === pattern.length
    ? [startPosition, textPosition]
    : null;
}

// Highlighting functions
function findMatchesForPatterns(
  text: string,
  patterns: string[],
  matchId: string,
): [number, number, string][] {
  return patterns.flatMap((pattern) =>
    findMatches(text, pattern).map(
      (match) => [...match, matchId] as [number, number, string],
    ),
  );
}

function generateUniqueId(): string {
  return `match-${Math.random().toString(36).slice(2, 11)}`;
}

interface PatternObject {
  match: string;
  color?: string;
}

function highlight(
  target: string | HTMLPreElement | NodeListOf<HTMLPreElement>,
  patterns: string | (string | PatternObject)[] | PatternObject,
  options: HighlightOptions = {},
  colorScheme: string[] = DefaultColors,
): void {
  if (!patterns || (Array.isArray(patterns) && patterns.length === 0)) {
    return;
  }

  const patternsArray = Array.isArray(patterns) ? patterns : [patterns];
  const elements =
    typeof target === "string"
      ? document.querySelectorAll<HTMLPreElement>(target)
      : target instanceof HTMLElement
        ? [target]
        : target;

  const { matchId = generateUniqueId() } = options;

  const processedPatterns = patternsArray.map((pattern, index) => {
    if (typeof pattern === "string") {
      return { match: pattern, color: colorScheme[index % colorScheme.length] };
    }
    return {
      match: pattern.match,
      color: pattern.color || colorScheme[index % colorScheme.length],
    };
  });

  elements.forEach((element) => {
    const text = element.textContent || "";

    const allMatches = processedPatterns.flatMap((pattern, index) => {
      const subPatterns = pattern.match
        .split(",")
        .map((p) => p.trim())
        .filter((p) => p !== "");
      return findMatchesForPatterns(
        text,
        subPatterns,
        `${matchId}-${index}`,
      ).map(
        (match) =>
          [...match, `--bylight-color: ${pattern.color};`] as [
            number,
            number,
            string,
            string,
          ],
      );
    });

    if (allMatches.length > 0) {
      element.innerHTML = `<code>${applyHighlights(text, allMatches)}</code>`;
    }
  });
}

function applyHighlights(
  text: string,
  matches: [number, number, string, string][],
): string {
  // Sort matches in reverse order based on their start index
  matches.sort((a, b) => b[0] - a[0]);

  return matches.reduce((result, [start, end, matchId, styleString]) => {
    const beforeMatch = result.slice(0, start);
    const matchContent = result.slice(start, end);
    const afterMatch = result.slice(end);

    return (
      beforeMatch +
      `<span class="bylight-code" style="${styleString}" data-match-id="${matchId}">` +
      matchContent +
      "</span>" +
      afterMatch
    );
  }, text);
}

// Link processing and hover effect functions
// Simplify the processLinksAndHighlight function
function processLinksAndHighlight(
  targetElement: HTMLElement,
  colorScheme: string[] = DefaultColors,
): void {
  // Clear any existing highlights in pre elements
  targetElement.querySelectorAll("pre code").forEach((codeElement) => {
    const preElement = codeElement.parentElement as HTMLPreElement;
    if (preElement && codeElement.innerHTML.includes("bylight-code")) {
      preElement.textContent = preElement.textContent || "";
    }
  });

  // Include both new links and already-processed spans
  const elements = targetElement.querySelectorAll<
    HTMLPreElement | HTMLAnchorElement | HTMLSpanElement
  >('pre, a[href^="bylight"], span.bylight-link');

  const preMap = new Map<HTMLPreElement, [number, number, string][]>();
  const linkMap = new Map<HTMLAnchorElement | HTMLSpanElement, LinkData>();
  const colorMap = new Map<string, number>();
  let colorIndex = 0;

  // Process all elements
  elements.forEach((element, index) => {
    if (element.tagName === "PRE") {
      preMap.set(element as HTMLPreElement, []);
    } else if (element.tagName === "A") {
      const anchorElement = element as HTMLAnchorElement;
      const linkData = processAnchorElement(
        anchorElement,
        index,
        colorScheme,
        colorIndex,
      );
      linkMap.set(anchorElement, linkData);
      colorMap.set(linkData.matchId, colorIndex);
      colorIndex = (colorIndex + 1) % colorScheme.length;
    } else if (
      element.tagName === "SPAN" &&
      element.classList.contains("bylight-link")
    ) {
      const spanElement = element as HTMLSpanElement;
      const linkData = processSpanElement(spanElement, index);
      linkMap.set(spanElement, linkData);
      colorMap.set(linkData.matchId, colorIndex);
      colorIndex = (colorIndex + 1) % colorScheme.length;
    }
  });

  // Second pass: Process links and find matches in pre elements
  linkMap.forEach(
    ({ targetIndices, patterns, index, matchId, color }, linkElement) => {
      const findMatchingPres = (
        indices: number[] | "all" | "up" | "down",
        index: number,
      ): HTMLPreElement[] => {
        if (indices === "all") {
          return Array.from(preMap.keys());
        }
        if (indices === "up" || indices === "down") {
          return findPreElementsInDirection(
            elements,
            index,
            indices,
            parseInt(indices),
          );
        }
        return indices
          .map((offset) => findPreElementByOffset(elements, index, offset))
          .filter((el): el is HTMLPreElement => el !== null);
      };

      const matchingPres = findMatchingPres(targetIndices, index);

      matchingPres.forEach((matchingPre) => {
        const text = matchingPre.textContent || "";
        const newMatches = findMatchesForPatterns(text, patterns, matchId);
        preMap.get(matchingPre)?.push(...newMatches);
      });
    },
  );

  // Apply highlights to pre elements
  preMap.forEach((matches, preElement) => {
    if (matches.length > 0) {
      const allMatches = matches.map(([start, end, matchId]) => {
        const linkData = Array.from(linkMap.values()).find(
          (data) => data.matchId === matchId,
        );
        const color =
          linkData?.color || colorScheme[colorMap.get(matchId) || 0];
        return [start, end, matchId, `--bylight-color: ${color};`] as [
          number,
          number,
          string,
          string,
        ];
      });
      preElement.innerHTML = `<code>${applyHighlights(preElement.textContent || "", allMatches)}</code>`;
      // preElement.bylightRefresh = () => preElement.innerHTML = `<code>${applyHighlights(preElement.textContent || "", allMatches)}</code>`;
      // preElement.bylightRefresh()
    }
  });

  // Process links
  linkMap.forEach((linkData, linkElement) => {
    const { matchId, color, targetIndices, patterns } = linkData;
    const finalColor = color || colorScheme[colorMap.get(matchId) || 0];

    // Only process if it's an anchor (not already a span)
    if (linkElement.tagName === "A") {
      // Create a new span element
      const spanElement = document.createElement("span");
      spanElement.innerHTML = linkElement.innerHTML;
      spanElement.dataset.matchId = matchId;
      spanElement.dataset.patterns = JSON.stringify(patterns);
      spanElement.dataset.targetIndices = JSON.stringify(targetIndices);
      spanElement.dataset.color = finalColor;
      spanElement.classList.add("bylight-link");
      spanElement.style.setProperty("--bylight-color", finalColor);

      // Replace the link with the span
      linkElement.parentNode?.replaceChild(spanElement, linkElement);
    }
  });
}

// Add a helper function to process anchor elements
function processAnchorElement(
  anchorElement: HTMLAnchorElement,
  index: number,
  colorScheme: string[],
  colorIndex: number,
): LinkData {
  const url = new URL(anchorElement.href);
  const matchId = generateUniqueId();
  const inParam = url.searchParams.get("in");
  const dirParam = url.searchParams.get("dir");
  const color = url.searchParams.get("color") || colorScheme[colorIndex];

  const targetIndices = getTargetIndices(inParam, dirParam);

  anchorElement.addEventListener("click", (e) => e.preventDefault());

  return {
    targetIndices,
    patterns: (
      url.searchParams.get("match") ||
      anchorElement.textContent ||
      ""
    ).split(","),
    index,
    matchId,
    color,
  };
}

// Add a helper function to process span elements (already processed links)
function processSpanElement(
  spanElement: HTMLSpanElement,
  index: number,
): LinkData {
  const patterns = JSON.parse(spanElement.dataset.patterns || "[]");
  const targetIndices = JSON.parse(spanElement.dataset.targetIndices || "[]");
  const matchId = spanElement.dataset.matchId || generateUniqueId();
  const color = spanElement.dataset.color || "";

  return {
    targetIndices,
    patterns,
    index,
    matchId,
    color,
  };
}

function getTargetIndices(
  inParam: string | null,
  dirParam: string | null,
): number[] | "all" | "up" | "down" {
  if (inParam) {
    return inParam === "all" ? "all" : inParam.split(",").map(Number);
  } else if (dirParam) {
    return dirParam as "up" | "down";
  }
  return [1]; // Default behavior
}

interface LinkData {
  targetIndices: number[] | "all" | "up" | "down";
  patterns: string[];
  index: number;
  matchId: string;
  color: string;
}

// Add these helper functions
function findPreElementsInDirection(
  elements: NodeListOf<HTMLPreElement | HTMLAnchorElement>,
  startIndex: number,
  direction: "up" | "down",
  count: number,
): HTMLPreElement[] {
  const dir = direction === "up" ? -1 : 1;
  const matchingPres: HTMLPreElement[] = [];
  let preCount = 0;

  for (let i = startIndex + dir; i >= 0 && i < elements.length; i += dir) {
    if (elements[i].tagName === "PRE") {
      matchingPres.push(elements[i] as HTMLPreElement);
      preCount++;
      if (preCount === count) break;
    }
  }

  return matchingPres;
}

function findPreElementByOffset(
  elements: NodeListOf<HTMLPreElement | HTMLAnchorElement>,
  startIndex: number,
  offset: number,
): HTMLPreElement | null {
  let preCount = 0;
  const dir = Math.sign(offset);

  for (let i = startIndex + dir; i >= 0 && i < elements.length; i += dir) {
    if (elements[i].tagName === "PRE") {
      preCount++;
      if (preCount === Math.abs(offset)) {
        return elements[i] as HTMLPreElement;
      }
    }
  }

  return null;
}

function addHoverEffect(targetElement: HTMLElement): void {
  targetElement.addEventListener("mouseover", (event) => {
    const target = event.target as HTMLElement;
    if (target.dataset.matchId) {
      const matchId = target.dataset.matchId;
      const elements = targetElement.querySelectorAll<HTMLElement>(
        `[data-match-id="${matchId}"]`,
      );
      elements.forEach((el) => {
        el.classList.add("bylight-hover");
      });
    }
  });

  targetElement.addEventListener("mouseout", (event) => {
    const target = event.target as HTMLElement;
    if (target.dataset.matchId) {
      const matchId = target.dataset.matchId;
      const elements = targetElement.querySelectorAll<HTMLElement>(
        `[data-match-id="${matchId}"]`,
      );
      elements.forEach((el) => {
        el.classList.remove("bylight-hover");
      });
    }
  });
}

// Main bylight function
/**
 * Initializes bylight syntax highlighting and link processing.
 * @param options Configuration options for bylight
 */
function bylight(options: BylightOptions = {}): void {
  const { target = "body", colorScheme = DefaultColors } = options;

  const targetElement =
    typeof target === "string"
      ? document.querySelector<HTMLElement>(target)
      : target;

  if (!targetElement) {
    console.error(`bylight: Target element not found - ${target}`);
    return;
  }

  processLinksAndHighlight(targetElement, colorScheme);
  addHoverEffect(targetElement);
}

bylight.highlight = highlight;
bylight.processLinksAndHighlight = processLinksAndHighlight;
bylight.addHoverEffect = addHoverEffect;
bylight.findMatches = findMatches;
bylight.findRegexMatches = findRegexMatches;
bylight.escapeRegExp = escapeRegExp;
bylight.DefaultColors = DefaultColors;

export {
  bylight as default,
  BylightOptions,
  HighlightOptions,
  PatternObject,
  highlight,
  processLinksAndHighlight,
  addHoverEffect,
  findMatches,
  findRegexMatches,
  escapeRegExp,
  DefaultColors,
};
