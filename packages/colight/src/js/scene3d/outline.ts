/**
 * Outline rendering system for Scene3D.
 *
 * This module contains types and pure functions for computing which elements
 * should be outlined and with what styles. The actual GPU rendering is still
 * handled by impl3d.tsx.
 */

import { ComponentConfig } from "./components";
import { HoverProps } from "./types";

// ============================================================================
// Types
// ============================================================================

export type OutlineConfig = {
  outline?: boolean;
  outlineColor?: [number, number, number];
  outlineWidth?: number;
};

export type OutlineTarget =
  | { kind: "component"; componentIdx: number }
  | { kind: "element"; componentIdx: number; elementIdx: number };

export type OutlineStyle = {
  color: [number, number, number];
  width: number;
};

export type OutlineGroup = {
  style: OutlineStyle;
  targets: OutlineTarget[];
};

// ============================================================================
// Constants
// ============================================================================

export const DEFAULT_OUTLINE_COLOR: [number, number, number] = [1, 1, 1];
export const DEFAULT_OUTLINE_WIDTH = 2;

// ============================================================================
// Functions
// ============================================================================

/**
 * Check if a config has any outline-related properties defined.
 */
export function hasOutlineConfig(config?: OutlineConfig): boolean {
  return (
    !!config &&
    (config.outline !== undefined ||
      config.outlineColor !== undefined ||
      config.outlineWidth !== undefined)
  );
}

/**
 * Check if outline is actually enabled for a config.
 * An outline is enabled if:
 * - `outline` is explicitly true, OR
 * - `outline` is undefined but `outlineColor` or `outlineWidth` is set
 */
export function isOutlineEnabled(config?: OutlineConfig): boolean {
  if (!config) return false;
  if (config.outline !== undefined) return config.outline;
  return config.outlineColor !== undefined || config.outlineWidth !== undefined;
}

/**
 * Resolve the outline style for a config, optionally falling back to a base config.
 */
export function resolveOutlineStyle(
  config?: OutlineConfig,
  fallback?: OutlineConfig,
): {
  enabled: boolean;
  color: [number, number, number];
  width: number;
} {
  const enabled = isOutlineEnabled(config ?? fallback);
  if (!enabled) {
    return {
      enabled: false,
      color: DEFAULT_OUTLINE_COLOR,
      width: DEFAULT_OUTLINE_WIDTH,
    };
  }

  return {
    enabled: true,
    color:
      config?.outlineColor ?? fallback?.outlineColor ?? DEFAULT_OUTLINE_COLOR,
    width:
      config?.outlineWidth ?? fallback?.outlineWidth ?? DEFAULT_OUTLINE_WIDTH,
  };
}

/**
 * Resolve outline style for a component's base configuration.
 */
export function resolveComponentOutline(component?: ComponentConfig) {
  return resolveOutlineStyle(component);
}

/**
 * Resolve outline style for a component's hover state.
 */
export function resolveHoverOutline(component?: ComponentConfig) {
  if (!component) {
    return {
      enabled: false,
      color: DEFAULT_OUTLINE_COLOR,
      width: DEFAULT_OUTLINE_WIDTH,
    };
  }
  if (!hasOutlineConfig(component.hoverProps)) {
    return {
      enabled: false,
      color: DEFAULT_OUTLINE_COLOR,
      width: DEFAULT_OUTLINE_WIDTH,
    };
  }
  return resolveOutlineStyle(component.hoverProps, component);
}

/**
 * Generate a unique key for an outline style (for grouping targets with same style).
 */
export function outlineStyleKey(
  color: [number, number, number],
  width: number,
): string {
  return `${color[0]}:${color[1]}:${color[2]}:${width}`;
}

/**
 * Check if two outline styles are identical.
 */
export function sameOutlineStyle(
  a: { color: [number, number, number]; width: number },
  b: { color: [number, number, number]; width: number },
): boolean {
  return (
    a.width === b.width &&
    a.color[0] === b.color[0] &&
    a.color[1] === b.color[1] &&
    a.color[2] === b.color[2]
  );
}

/**
 * Collect all outline targets grouped by their style.
 *
 * This is the main function that determines what should be outlined:
 * - Component-level outlines (always-on)
 * - Decoration-level outlines (always-on for specific elements)
 * - Group-level hover outlines (when group is hovered)
 * - Element-level hover outlines (when specific element is hovered)
 */
export function collectOutlineGroups(
  components: ComponentConfig[],
  hoverState: { componentIdx: number; elementIdx: number } | null,
  hoveredGroups?: Map<string, HoverProps>,
): OutlineGroup[] {
  const groups = new Map<string, OutlineGroup>();

  components.forEach((component, componentIdx) => {
    // Component-level outlines
    const outline = resolveComponentOutline(component);
    if (outline.enabled) {
      const key = outlineStyleKey(outline.color, outline.width);
      const existing = groups.get(key);
      const target: OutlineTarget = { kind: "component", componentIdx };
      if (existing) {
        existing.targets.push(target);
      } else {
        groups.set(key, {
          style: { color: outline.color, width: outline.width },
          targets: [target],
        });
      }
    }

    // Decoration-level outlines
    if (component.decorations) {
      for (const decoration of component.decorations) {
        const decoOutline = resolveOutlineStyle(decoration, component);
        if (!decoOutline.enabled) continue;

        const key = outlineStyleKey(decoOutline.color, decoOutline.width);
        for (const elementIdx of decoration.indexes) {
          const existing = groups.get(key);
          const target: OutlineTarget = {
            kind: "element",
            componentIdx,
            elementIdx,
          };
          if (existing) {
            existing.targets.push(target);
          } else {
            groups.set(key, {
              style: { color: decoOutline.color, width: decoOutline.width },
              targets: [target],
            });
          }
        }
      }
    }

    // Group-level hover outlines (apply to entire component when group is hovered)
    if (hoveredGroups && hoveredGroups.size > 0) {
      const groupPath = (component as any)._groupPath as string[] | undefined;
      if (groupPath?.length) {
        // Check each ancestor group for outline hoverProps
        for (let i = groupPath.length; i > 0; i--) {
          const ancestorPath = groupPath.slice(0, i).join("/");
          const hoverProps = hoveredGroups.get(ancestorPath);
          if (hoverProps?.outline) {
            const outlineColor =
              hoverProps.outlineColor ?? DEFAULT_OUTLINE_COLOR;
            const outlineWidth =
              hoverProps.outlineWidth ?? DEFAULT_OUTLINE_WIDTH;
            const key = outlineStyleKey(outlineColor, outlineWidth);
            const existing = groups.get(key);
            const target: OutlineTarget = { kind: "component", componentIdx };
            if (existing) {
              // Avoid duplicates
              if (
                !existing.targets.some(
                  (t) =>
                    t.kind === "component" && t.componentIdx === componentIdx,
                )
              ) {
                existing.targets.push(target);
              }
            } else {
              groups.set(key, {
                style: { color: outlineColor, width: outlineWidth },
                targets: [target],
              });
            }
            break; // Only apply innermost group's outline
          }
        }
      }
    }
  });

  if (hoverState) {
    const hoveredComponent = components[hoverState.componentIdx];
    const hoverOutline = resolveHoverOutline(hoveredComponent);
    if (hoverOutline.enabled) {
      const baseOutline = resolveComponentOutline(hoveredComponent);
      const shouldAdd =
        !baseOutline.enabled || !sameOutlineStyle(baseOutline, hoverOutline);
      if (shouldAdd) {
        const key = outlineStyleKey(hoverOutline.color, hoverOutline.width);
        const existing = groups.get(key);
        const target: OutlineTarget = {
          kind: "element",
          componentIdx: hoverState.componentIdx,
          elementIdx: hoverState.elementIdx,
        };
        if (existing) {
          existing.targets.push(target);
        } else {
          groups.set(key, {
            style: {
              color: hoverOutline.color,
              width: hoverOutline.width,
            },
            targets: [target],
          });
        }
      }
    }
  }

  return Array.from(groups.values());
}
