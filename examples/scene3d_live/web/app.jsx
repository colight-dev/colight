import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Scene, DEFAULT_CAMERA, screenRay } from "./scene3d.mjs";
import { packMessage, unpackMessage } from "./wire_protocol.js";

const WS_URL = `ws://${window.location.hostname}:8001`;

function intersectPlane(ray, planePoint, planeNormal) {
  if (!ray) return null;
  const denom =
    ray.direction[0] * planeNormal[0] +
    ray.direction[1] * planeNormal[1] +
    ray.direction[2] * planeNormal[2];
  if (Math.abs(denom) < 1e-6) return null;
  const t =
    ((planePoint[0] - ray.origin[0]) * planeNormal[0] +
      (planePoint[1] - ray.origin[1]) * planeNormal[1] +
      (planePoint[2] - ray.origin[2]) * planeNormal[2]) /
    denom;
  if (t < 0) return null;
  return [
    ray.origin[0] + ray.direction[0] * t,
    ray.origin[1] + ray.direction[1] * t,
    ray.origin[2] + ray.direction[2] * t,
  ];
}

function useWireScene(url) {
  const [scene, setScene] = useState(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);
  const pendingRef = useRef(null);

  useEffect(() => {
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    ws.onmessage = (event) => {
      if (typeof event.data === "string") {
        const envelope = JSON.parse(event.data);
        if (envelope.buffer_count > 0) {
          pendingRef.current = { envelope, buffers: [] };
          return;
        }
        const payload = unpackMessage(envelope, []);
        if (payload?.scene) {
          setScene(payload.scene);
        }
        return;
      }

      if (!pendingRef.current) {
        return;
      }
      pendingRef.current.buffers.push(event.data);
      const { envelope, buffers } = pendingRef.current;
      if (buffers.length === envelope.buffer_count) {
        pendingRef.current = null;
        const payload = unpackMessage(envelope, buffers);
        if (payload?.scene) {
          setScene(payload.scene);
        }
      }
    };

    return () => ws.close();
  }, [url]);

  const sendEvent = (eventName, data) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const [envelope, buffers] = packMessage({
      type: "event",
      event: eventName,
      data,
    });
    ws.send(JSON.stringify(envelope));
    buffers.forEach((buffer) => ws.send(buffer));
  };

  return { scene, connected, sendEvent };
}

function App() {
  const { scene, connected, sendEvent } = useWireScene(WS_URL);
  const [localScene, setLocalScene] = useState(scene);
  const [camera, setCamera] = useState(DEFAULT_CAMERA);
  const cameraRef = useRef(camera);
  const hoverRef = useRef(null);
  const dragRef = useRef(null);
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const sceneRef = useRef(localScene);
  const sendRef = useRef(sendEvent);
  const rafRef = useRef(null);
  const pendingPayloadRef = useRef(null);

  useEffect(() => {
    cameraRef.current = camera;
  }, [camera]);

  useEffect(() => {
    setLocalScene(scene);
  }, [scene]);

  useEffect(() => {
    sceneRef.current = localScene;
  }, [localScene]);

  useEffect(() => {
    sendRef.current = sendEvent;
  }, [sendEvent]);

  const renderedScene = useMemo(() => {
    if (!localScene) return null;
    const components = localScene.components.map((component, idx) => {
      const existingHover = component.onHover;
      const existingHoverDetail = component.onHoverDetail;
      return {
        ...component,
        onHover: (elementIdx) => {
          if (typeof elementIdx === "number") {
            hoverRef.current = {
              componentIndex: idx,
              elementIndex: elementIdx,
            };
          } else {
            hoverRef.current = null;
          }
          if (typeof existingHover === "function") {
            existingHover(elementIdx);
          }
        },
        onHoverDetail: (info) => {
          if (info) {
            hoverRef.current = {
              componentIndex: idx,
              elementIndex: info.instanceIndex,
              face: info.face,
              hit: info.hit,
              ray: info.ray,
              camera: info.camera,
              screen: info.screen,
            };
          }
          if (typeof existingHoverDetail === "function") {
            existingHoverDetail(info);
          }
        },
      };
    });
    return { ...localScene, components };
  }, [localScene]);

  useEffect(() => {
    if (renderedScene?.defaultCamera) {
      setCamera(renderedScene.defaultCamera);
    }
  }, [renderedScene?.defaultCamera]);

  const scheduleSend = (eventName, payload) => {
    pendingPayloadRef.current = { eventName, payload };
    if (rafRef.current) return;
    rafRef.current = requestAnimationFrame(() => {
      const queued = pendingPayloadRef.current;
      pendingPayloadRef.current = null;
      rafRef.current = null;
      if (queued) {
        sendRef.current(queued.eventName, queued.payload);
      }
    });
  };

  const updateCenter = (componentIndex, elementIndex, position) => {
    setLocalScene((prev) => {
      if (!prev) return prev;
      const components = [...prev.components];
      const component = components[componentIndex];
      if (!component?.centers) return prev;
      const centers = component.centers;
      const nextCenters =
        centers instanceof Float32Array
          ? centers.slice()
          : Float32Array.from(centers);
      const base = elementIndex * 3;
      nextCenters[base] = position[0];
      nextCenters[base + 1] = position[1];
      nextCenters[base + 2] = position[2];
      components[componentIndex] = { ...component, centers: nextCenters };
      return { ...prev, components };
    });
  };

  const startDrag = (event) => {
    if (!containerRef.current || !sceneRef.current) return;
    const hover = hoverRef.current;
    if (!hover) return;
    const component = sceneRef.current.components[hover.componentIndex];
    const dragToken = component?.drag?.__event__;
    if (!dragToken || !component?.centers) return;

    event.preventDefault();
    event.stopPropagation();

    // Get the current center
    const centers = component.centers;
    const base = hover.elementIndex * 3;
    const originalCenter = [
      centers[base + 0],
      centers[base + 1],
      centers[base + 2],
    ];

    // The hit point on the surface from the GPU
    if (!hover.hit?.position) {
      console.error("No hit position from picking system!");
      return;
    }
    const hitPoint = [hover.hit.position[0], hover.hit.position[1], hover.hit.position[2]];

    // Use the normal from the hit point - must exist!
    if (!hover.hit?.normal) {
      console.error("No hit normal from picking system!");
      return;
    }
    const planeNormal = [hover.hit.normal[0], hover.hit.normal[1], hover.hit.normal[2]];

    // The drag plane passes through the hit point (on the surface).
    // We'll track the offset from intersection to the object's center.
    const dragPlanePoint = hitPoint;

    // Compute a fresh ray from the ACTUAL click position
    // Use the CANVAS rect (not container), same as GPU picking uses
    const rect = canvasRef.current?.getBoundingClientRect() ?? containerRef.current.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const clickY = event.clientY - rect.top;
    const pickingRay = screenRay(clickX, clickY, rect, cameraRef.current);

    const initialCursorOnPlane = intersectPlane(pickingRay, dragPlanePoint, planeNormal);

    if (!initialCursorOnPlane) {
      console.error("Could not intersect ray with drag plane!");
      return;
    }

    dragRef.current = {
      componentIndex: hover.componentIndex,
      elementIndex: hover.elementIndex,
      originalCenter,
      initialCursorOnPlane,
      dragPlanePoint,
      planeNormal,
      eventName: dragToken,
    };

    window.addEventListener("pointermove", handleDragMove);
    window.addEventListener("pointerup", handleDragEnd, { once: true });
  };

  const handleDragMove = (event) => {
    if (!dragRef.current || !containerRef.current) return;
    // Use the CANVAS rect, same as GPU picking
    const rect = canvasRef.current?.getBoundingClientRect() ?? containerRef.current.getBoundingClientRect();
    const currentScreenX = event.clientX - rect.left;
    const currentScreenY = event.clientY - rect.top;

    // Cast ray from current mouse position using the proper library function
    const currentRay = screenRay(
      currentScreenX,
      currentScreenY,
      rect,
      cameraRef.current,
    );

    if (!currentRay) return;

    // Find where the cursor ray intersects the drag plane
    const cursorOnPlane = intersectPlane(
      currentRay,
      dragRef.current.dragPlanePoint,
      dragRef.current.planeNormal,
    );

    if (!cursorOnPlane) {
      return;
    }

    // Compute delta from initial CPU intersection to current CPU intersection
    // Both use the same ray computation, so the delta is consistent
    const deltaX = cursorOnPlane[0] - dragRef.current.initialCursorOnPlane[0];
    const deltaY = cursorOnPlane[1] - dragRef.current.initialCursorOnPlane[1];
    const deltaZ = cursorOnPlane[2] - dragRef.current.initialCursorOnPlane[2];

    const newCenter = [
      dragRef.current.originalCenter[0] + deltaX,
      dragRef.current.originalCenter[1] + deltaY,
      dragRef.current.originalCenter[2] + deltaZ,
    ];

    updateCenter(
      dragRef.current.componentIndex,
      dragRef.current.elementIndex,
      newCenter,
    );
    scheduleSend(dragRef.current.eventName, {
      index: dragRef.current.elementIndex,
      position: newCenter,
    });
  };

  const handleDragEnd = () => {
    dragRef.current = null;
    window.removeEventListener("pointermove", handleDragMove);
  };

  if (!renderedScene) {
    return (
      <div
        style={{
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          letterSpacing: "0.08em",
          textTransform: "uppercase",
        }}
      >
        Connecting...
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      style={{ width: "100%", height: "100%" }}
      onPointerDownCapture={startDrag}
    >
      <Scene
        components={renderedScene.components}
        defaultCamera={renderedScene.defaultCamera || DEFAULT_CAMERA}
        controls={renderedScene.controls}
        onCameraChange={setCamera}
        onCanvasRef={(canvas) => { canvasRef.current = canvas; }}
      />
      <div
        style={{
          position: "absolute",
          top: 16,
          left: 16,
          padding: "8px 12px",
          background: "rgba(15, 23, 42, 0.7)",
          borderRadius: 8,
          fontSize: 12,
        }}
      >
        {connected ? "Drag the cube" : "Connecting..."}
      </div>
    </div>
  );
}

const root = createRoot(document.getElementById("root"));
root.render(<App />);
