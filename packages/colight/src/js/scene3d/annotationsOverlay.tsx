/**
 * @module annotationsOverlay
 * @description DOM overlay that renders named annotation callouts over a
 * scene3d canvas — the annotation analogue of the legend overlay (legend.tsx).
 *
 * Each callout is a marker dot at its projected anchor, a thin leader line to a
 * fixed screen-offset label, and the label itself. The overlay is
 * pointer-events:none (it must never intercept canvas interaction) and is part
 * of the captured DOM, so it appears in the same full-page screenshots the
 * legend does. A hidden data-colight-annotations marker mirrors the resolved
 * callouts (name, text, anchor, world, screen, visible) for screenshot --json.
 *
 * Anchors behind the camera or outside the viewport are hidden (not clamped to
 * the edge, per the v1 design) and reported visible:false.
 */

import React from "react";
import { tw } from "../utils";
import { ComponentConfig, PrimitiveSpec } from "./components";
import { GPUTransform } from "./gpu-transforms";
import { CameraState } from "./camera3d";
import {
  Annotations,
  resolveAnnotations,
  ResolvedAnnotation,
} from "./annotations";

/** Fixed leader offset from the marker to the label (screen px, up-and-right). */
const LEADER_DX = 22;
const LEADER_DY = -22;

interface AnnotationsOverlayProps {
  annotations: Annotations;
  components: ComponentConfig[];
  specFor: (component: ComponentConfig) => PrimitiveSpec<any> | undefined;
  transforms: GPUTransform[];
  camera: CameraState;
  /** Canvas rect: left/top are the overlay's own origin (0,0 here); width/
   * height bound the viewport for the visibility test. */
  rect: { left: number; top: number; width: number; height: number };
  origin?: [number, number, number] | null;
}

function cssColor(rgb: [number, number, number]): string {
  return `rgb(${Math.round(rgb[0] * 255)}, ${Math.round(rgb[1] * 255)}, ${Math.round(
    rgb[2] * 255,
  )})`;
}

/** Serialized per-annotation payload for the hidden reporting marker. */
function annotationReport(a: ResolvedAnnotation): Record<string, unknown> {
  return {
    name: a.name,
    text: a.text,
    anchor: a.anchor,
    world: a.world,
    // Canvas-local CSS pixels (origin = canvas top-left). screenshot --json
    // offsets these by the scene rect to land in page pick-at pixel space.
    screen: a.screen,
    visible: a.visible,
  };
}

/**
 * One callout: marker dot + leader line + label. Rendered only when the anchor
 * is visible (in front of the camera and inside the viewport).
 */
function Callout({ a }: { a: ResolvedAnnotation }) {
  if (!a.visible || !a.screen) return null;
  const { x, y } = a.screen;
  const labelX = x + LEADER_DX;
  const labelY = y + LEADER_DY;
  const color = cssColor(a.color);
  return (
    <>
      {/* Leader line from marker to label anchor. */}
      <svg
        className={tw("absolute top-0 left-0 overflow-visible")}
        style={{ pointerEvents: "none" }}
        width={1}
        height={1}
      >
        <line
          x1={x}
          y1={y}
          x2={labelX}
          y2={labelY}
          stroke={color}
          strokeWidth={1}
        />
      </svg>
      {/* Marker dot at the anchor. */}
      <div
        className={tw("absolute rounded-full")}
        style={{
          left: x - 4,
          top: y - 4,
          width: 8,
          height: 8,
          background: color,
          border: "1.5px solid rgba(255,255,255,0.9)",
          boxShadow: "0 0 2px rgba(0,0,0,0.4)",
          pointerEvents: "none",
        }}
      />
      {/* Label, anchored bottom-left at the leader end. */}
      <div
        className={tw(
          "absolute px-[6px] py-[2px] text-[11px] leading-[1.3] text-gray-900 rounded-[3px] whitespace-nowrap",
        )}
        style={{
          left: labelX,
          top: labelY,
          transform: "translateY(-100%)",
          background: "rgba(255,255,255,0.9)",
          border: `1px solid ${color}`,
          boxShadow: "0 1px 3px rgba(0,0,0,0.15)",
          pointerEvents: "none",
        }}
      >
        {a.text}
      </div>
    </>
  );
}

/**
 * Projects every annotation with the current camera and renders the callouts
 * plus a hidden reporting marker. Absolutely positioned over the canvas,
 * pointer-events:none.
 */
export function AnnotationsOverlay({
  annotations,
  components,
  specFor,
  transforms,
  camera,
  rect,
  origin,
}: AnnotationsOverlayProps) {
  const resolved = resolveAnnotations(
    annotations,
    components,
    specFor,
    transforms,
    camera,
    { width: rect.width, height: rect.height },
    origin,
  );
  if (resolved.length === 0) return null;

  return (
    <div
      className={tw("absolute top-0 left-0 z-20 w-full h-full")}
      style={{ pointerEvents: "none" }}
      data-colight-annotations={JSON.stringify(resolved.map(annotationReport))}
    >
      {resolved.map((a) => (
        <Callout key={a.name} a={a} />
      ))}
    </div>
  );
}
