# Scene3D Feature Integration Plan

## Overview

This document tracks the integration of features from `genweb/scene3d-ext` into `genweb/scene3d-ext-2` (based on `genweb-scene3d-improvements`).

**Base branch**: `genweb-scene3d-improvements` - Has better primitive system, outline pass, prepared for publishing
**Feature branch**: `genweb/scene3d-ext` - Has drag, groups, overlays, hoverProps, detailed pick info

**Approach**: Feature-by-feature additions, NOT file merges. Use the feature branch as reference.

---

## Branch Differences Summary

### Base Branch [base] (`genweb-scene3d-improvements`) Strengths
1. **Declarative primitive system** (`primitives/define.ts`) - `definePrimitive()` with attribute schemas, auto-generated shaders
2. **Outline pass** - Post-process hover outline effect (outlineVertCode, outlineFragCode in shaders.ts). NOTE: We want to *keep* this pass but invoke it via a `hoverOutline` *prop* instead of directly linking it to hover - because then it can be invoked via the [feature] hoverProps system. The prop can either be null (not present) or a color.
3. **Scene3d independence** - No dependency on `$StateContext`, self-contained module. [feature] also did this, and went further by properly bringing in the canvasSnapshot system.
4. **Type-checked and publish-ready** - Clean exports, proper TypeScript
5. **BoundingBox primitive** - was added, using the new system

### Feature Branch [feature] (`genweb/scene3d-ext`) additions
- **hoverProps** - Declarative hover styling without state management
- **Drag system** - `onDragStart`, `onDrag`, `onDragEnd` callbacks with constraints. 
- **Overlays** - `layer: "overlay"` for gizmos that render in front
- **pickingScale** - Inflated picking geometry for easier clicking
- **Detailed pick info** - `onHoverDetail`, `onClickDetail` with ray, hit position, etc.
- **Gizmo utilities** - `createTranslateGizmo` for transform handles
- **Groups** - Hierarchical transforms via `Group` component. NOTE: we may want to do extra planning when integrating this. Not 100% sure about the design.
- **scene3d.tsx entrypoint** - A reorganized scene3d.tsx entry structure - this should be used - accepting either scene props or layers.

### Structural Differences

| Aspect | Base (improvements) | Feature (scene3d-ext) |
|--------|--------------------|-----------------------|
| Export name | `SceneInner` | `SceneImpl` |
| Utils location | `../utils` (outside scene3d) | `./utils` (inside scene3d) |
| canvasSnapshot | `../canvasSnapshot` (outside) | `./canvasSnapshot` (inside) |
| Context dependency | None (removed) | Uses `$StateContext` |
| Primitive registry | Not needed (uses primitives/) | `registry.ts` with `createSimplePrimitive` |
| Hover effect | Outline post-process pass | hoverProps declarative styling |

---

## Files Already Ported (on genweb/scene3d-ext-2)

These standalone files have been copied and should work:

- [x] `vec3.ts` - Vector math utilities
- [x] `ray.ts` - Ray/plane intersection utilities
- [x] `project.ts` - Screen projection utilities
- [x] `pointer.ts` - Pointer context types
- [x] `pick-info.ts` - Detailed pick information builder
- [x] `drag.ts` - Drag constraint resolution
- [x] `groups.ts` - Component grouping/transforms
- [x] `gizmo.ts` - Transform gizmo generation
- [x] `registry.ts` - Minimal primitive registry (just lookup, no createSimplePrimitive)
- [x] `index.ts` - Module exports (needs update after features added)

---

## Types Already Extended (in types.ts)

- [x] `ReadyState` and `NOOP_READY_STATE`
- [x] `arrayFields` on `PrimitiveSpec`
- [x] `HoverProps` interface
- [x] `RenderLayer` type ("scene" | "overlay")
- [x] `layer` on `BaseComponentConfig`
- [x] `onHoverDetail`, `onClickDetail` callbacks
- [x] `onDragStart`, `onDrag`, `onDragEnd` callbacks
- [x] `dragConstraint` on `BaseComponentConfig`
- [x] `hoverProps` on `BaseComponentConfig`
- [x] `pickingScale` on `BaseComponentConfig`
- [x] `PickEventType`, `DragConstraint`, `DragInfo`
- [x] `PickRay`, `PickHit`, `PickFace`, `PickSegment`, `PickCamera`, `PickInfo`
- [x] `targets` on `PipelineConfig`
- [x] `overlayPipeline`, `overlayPickingPipeline` on `RenderObject`
- [x] `layer` on `RenderObject`
- [x] Removed `BoundingBox` from `GeometryResources` (it's a primitive now)

---

## Components.ts Updates Done

- [x] `createOverlayPipeline` function added
- [x] `getLineBeamsSegmentPointIndex` exported from primitives/lineBeams.ts

---

## Tests Copied

- [x] `scene3d-groups.test.ts`
- [x] `scene3d-hover-props.test.ts`
- [x] `scene3d-pick-info.test.js`

---

## Features To Implement (in impl3d.tsx)

### Feature 1: hoverProps Support
**Reference**: `genweb/scene3d-ext` impl3d.tsx lines ~887-892, ~1471-1522, ~1931-1963, ~1997-2071

**What it does**: Allows declarative hover styling:
```tsx
Ellipsoid({
  centers: [[0, 0, 0]],
  color: [1, 0, 0],
  hoverProps: { color: [1, 0.5, 0.5], scale: 1.2 }
})
```

**Implementation**:
1. Add `hoverPropsState` ref: `Map<componentIdx, elementIdx>`
2. Add `hoverPropsDirty` ref for change tracking
3. In render loop, check if component has hoverProps and apply as decoration
4. On hover change, update hoverPropsState and request re-render
5. Pass `hoveredIndex` to `buildRenderData`

**Key code patterns**:
```typescript
// Track hover state per component
const hoverPropsState = useRef<Map<number, number>>(new Map());
const hoverPropsDirty = useRef(false);

// In buildRenderData call, pass hoveredIndex
const hoveredIndex = component.hoverProps
  ? hoverPropsState.current.get(offset.componentIdx)
  : undefined;
```

---

### Feature 2: pickingScale Support
**Reference**: `genweb/scene3d-ext` impl3d.tsx - buildPickingData calls

**What it does**: Inflates picking geometry for easier clicking on thin objects:
```tsx
LineBeams({
  points: [...],
  size: 0.02,
  pickingScale: 3.0  // 3x larger hit area
})
```

**Implementation**:
1. In `buildPickingData` calls, apply `pickingScale` to scale the geometry
2. Modify the spec's `applyDecorationScale` or handle in buildPickingData

---

### Feature 3: Overlay Rendering
**Reference**: `genweb/scene3d-ext` impl3d.tsx - pipeline creation, render pass

**What it does**: Components with `layer: "overlay"` render in front of scene geometry.

**Implementation**:
1. Create overlay pipelines (already have `createOverlayPipeline`)
2. In render pass, render scene layer first, then overlay layer
3. In pick pass, pick overlay first (priority), then scene
4. Track `layer` on RenderObject

---

### Feature 4: Drag System Integration
**Reference**: `genweb/scene3d-ext` impl3d.tsx - mouse event handlers, drag state
**What it does**: Full drag support with constraints:
```tsx
Ellipsoid({
  centers: positions,
  onDragStart: (info) => console.log('started', info),
  onDrag: (info) => setPositions(updatePosition(info)),
  onDragEnd: (info) => console.log('ended', info),
  dragConstraint: { type: 'plane', normal: [0, 1, 0] }
})
```

**Implementation**:
1. Add drag state tracking (isDragging, dragStartInfo, resolvedConstraint)
2. Import and use `resolveConstraint`, `computeConstrainedPosition` from drag.ts
3. In mouse handlers: detect drag start, compute constrained position, call callbacks
4. Handle pointer capture for smooth dragging

---

### Feature 5: Detailed Pick Info
**Reference**: `genweb/scene3d-ext` impl3d.tsx - buildPickInfo calls

**What it does**: Provides rich information about picks:
```tsx
onClickDetail: (info) => {
  console.log(info.ray, info.hit.position, info.camera);
}
```

**Implementation**:
1. Use `buildPickInfo` from pick-info.ts
2. Call in hover/click handlers alongside simple callbacks
3. Include camera state, ray, hit position, face info, etc.

---

### Feature 6: Groups Support
**Reference**: `genweb/scene3d-ext` scene3d.tsx - Group component, flattenGroups

**What it does**: Hierarchical transforms:
```tsx
Group({
  name: "arm",
  position: [1, 0, 0],
  children: [
    Ellipsoid({ centers: [[0, 0, 0]] }),
    Group({ name: "hand", position: [0.5, 0, 0], children: [...] })
  ]
})
```

**Implementation**:
1. In scene3d.tsx, use `flattenGroups` to flatten before passing to SceneInner
2. Apply accumulated transforms to component positions/rotations
3. Track groupPath for pick info

---

## Scene3d.tsx Changes Needed

1. **Naming**: Keep `SceneInner` (base branch naming)
2. **Group component**: Add `Group` export and `flattenGroups` usage
3. **Type coercion**: The base branch moved coercion into Scene, which is important for the declarative API

---

## Important Design Decisions

### 1. Outline Pass vs hoverProps
- Base branch has outline pass (post-process effect)
- Feature branch has hoverProps (inline attribute modification)
- **Decision**: Keep BOTH - they serve different purposes
  - hoverProps: Declarative color/scale/alpha changes
  - Outline: Visual edge highlight effect
- The two can coexist
- Change Outline to not be invoked on hover - "just a prop" - then it can be used VIA hoverProps

### 2. Utils Location
- Base branch: `../utils` (outside scene3d)
- Feature branch: `./utils` (inside scene3d)
- **Decision**: Keep outside (`../utils`) for the base branch structure
- This maintains scene3d independence while sharing utilities

### 3. Registry vs Primitives
- Base branch: `primitives/define.ts` with `definePrimitive()`
- Feature branch: `registry.ts` with `createSimplePrimitive()`
- **Decision**: Use base branch's `definePrimitive()` - it's more powerful
- The minimal registry.ts we created is just for runtime lookup

### 4. Context Dependency
- Base branch: Removed `$StateContext` dependency
- Feature branch: Uses `$StateContext`
- **Decision**: Keep scene3d context-free (base branch approach)
- This is important for publishing scene3d as an independent package

---

## Testing Strategy

Run after each feature:
```bash
yarn vitest run packages/colight/tests/js/scene3d
```

Key test files:
- `scene3d-hover-props.test.ts` - Tests hoverProps and pickingScale
- `scene3d-groups.test.ts` - Tests group transforms
- `scene3d-pick-info.test.js` - Tests detailed pick info

---

## File Reference

### impl3d.tsx Key Sections (feature branch line numbers)

- Imports: 1-80
- SceneImplProps interface: 82-160
- GPU state initialization: 430-620
- hoverProps state: 887-892
- buildRenderData with hoverProps: 1471-1530
- Pick pass: 1615-1870
- Hover handling with hoverProps: 1931-2071
- Drag handling: 2073-2200
- Mouse event handlers: 2400-2550

### scene3d.tsx Key Sections (feature branch)

- Group component: 150-180
- flattenGroups usage: 250-280
- Component type coercion: 300-350

---

## Next Steps

### Feature 1: hoverProps support
1. [ ] Update `buildRenderData` in components.ts to accept `hoveredIndex` parameter
2. [ ] Add hoverProps application logic after decorations
3. [ ] Add `hoverPropsState` ref and dirty tracking in impl3d.tsx
4. [ ] Pass hoveredIndex to buildRenderData calls
5. [ ] On hover change, update hoverPropsState and request re-render
6. [ ] Run hoverProps tests

### Feature 2: pickingScale support
1. [ ] Update `buildPickingData` to apply `pickingScale` to geometry
2. [ ] Run pickingScale tests

### Feature 3: Outline as component-level prop
1. [ ] Add `hoverOutline` prop to BaseComponentConfig (default: undefined)
2. [ ] Add `defaultHoverOutline` prop to Scene for scene-level default
3. [ ] Update impl3d to check component.hoverOutline when rendering outline pass
4. [ ] Update silhouette rendering to only outline components with hoverOutline enabled

### Feature 4: Overlay rendering
1. [ ] Create overlay pipelines using existing `createOverlayPipeline`
2. [ ] Separate render objects by layer (scene vs overlay)
3. [ ] Render scene layer first, then overlay layer
4. [ ] Pick overlay layer first (priority), then scene layer

### Feature 5: Drag system
1. [ ] Add drag state tracking refs in impl3d.tsx
2. [ ] Import `resolveConstraint`, `computeConstrainedPosition` from drag.ts
3. [ ] In mouse handlers: detect drag start, compute constrained position
4. [ ] Call drag callbacks (onDragStart, onDrag, onDragEnd)
5. [ ] Handle pointer capture

### Feature 6: Detailed pick info
1. [ ] Import `buildPickInfo` from pick-info.ts
2. [ ] Call buildPickInfo in hover/click handlers
3. [ ] Pass info to onHoverDetail, onClickDetail callbacks

### Feature 7: Groups support
1. [ ] Add `Group` export to scene3d.tsx
2. [ ] Use `flattenGroups` in Scene before passing to SceneInner
3. [ ] Track groupPath in pick info

### Final steps
1. [ ] Update index.ts exports (verify all exports work)
2. [ ] Run full test suite
3. [ ] Port Python bindings for new features

---

## Git Commands for Reference

```bash
# View specific file from feature branch
git show genweb/scene3d-ext:packages/colight/src/js/scene3d/impl3d.tsx | head -100

# Diff between branches for a file
git diff genweb-scene3d-improvements genweb/scene3d-ext -- packages/colight/src/js/scene3d/impl3d.tsx

# View specific function in feature branch
git show genweb/scene3d-ext:packages/colight/src/js/scene3d/impl3d.tsx | grep -A 50 "hoverPropsState"
```
