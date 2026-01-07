import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Scene,
  DEFAULT_CAMERA,
  screenRay,
  intersectPlane,
  planeFromHit,
  sub,
  add,
} from "@colight/scene3d";
import { packMessage, unpackMessage } from "@colight/wire-protocol";

const WS_URL = `ws://${window.location.hostname}:8001`;

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
  const [camera, setCamera] = useState(DEFAULT_CAMERA);
  const pointerRef = useRef(null);
  const dragRef = useRef(null);
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const sceneRef = useRef(scene);
  const sendRef = useRef(sendEvent);
  const rafRef = useRef(null);
  const pendingPayloadRef = useRef(null);

  useEffect(() => {
    sceneRef.current = scene;
  }, [scene]);

  useEffect(() => {
    sendRef.current = sendEvent;
  }, [sendEvent]);

  useEffect(() => {
    if (scene?.defaultCamera) {
      setCamera(scene.defaultCamera);
    }
  }, [scene?.defaultCamera]);

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

  const startDrag = (event) => {
    // Use pointer context from Scene
    const pointer = pointerRef.current;
    if (!pointer?.pick?.hit) return;

    const { pick, rect, camera } = pointer;
    const component = sceneRef.current?.components[pick.component.index];
    const dragToken = component?.drag?.__event__;
    if (!dragToken || !component?.centers) return;

    event.preventDefault();
    event.stopPropagation();

    // Get the current center
    const centers = component.centers;
    const base = pick.instanceIndex * 3;
    const originalCenter = [
      centers[base + 0],
      centers[base + 1],
      centers[base + 2],
    ];

    // Create drag plane from hit data using utility
    const plane = planeFromHit(pick.hit);
    if (!plane) {
      console.error("Could not create drag plane from hit!");
      return;
    }

    // Get initial cursor position on plane
    const clickX = event.clientX - containerRef.current.getBoundingClientRect().left;
    const clickY = event.clientY - containerRef.current.getBoundingClientRect().top;
    const ray = screenRay(clickX, clickY, rect, camera);
    const initialCursorOnPlane = intersectPlane(ray, plane);

    if (!initialCursorOnPlane) {
      console.error("Could not intersect ray with drag plane!");
      return;
    }

    dragRef.current = {
      componentIndex: pick.component.index,
      elementIndex: pick.instanceIndex,
      originalCenter,
      initialCursorOnPlane,
      plane,
      eventName: dragToken,
      rect,
      camera,
    };

    window.addEventListener("pointermove", handleDragMove);
    window.addEventListener("pointerup", handleDragEnd, { once: true });
  };

  const handleDragMove = (event) => {
    const drag = dragRef.current;
    if (!drag || !containerRef.current) return;

    const containerRect = containerRef.current.getBoundingClientRect();
    const currentScreenX = event.clientX - containerRect.left;
    const currentScreenY = event.clientY - containerRect.top;

    // Cast ray from current mouse position
    const currentRay = screenRay(currentScreenX, currentScreenY, drag.rect, drag.camera);
    if (!currentRay) return;

    // Find where the cursor ray intersects the drag plane
    const cursorOnPlane = intersectPlane(currentRay, drag.plane);
    if (!cursorOnPlane) return;

    // Compute new center: original + (current - initial)
    const delta = sub(cursorOnPlane, drag.initialCursorOnPlane);
    const newCenter = add(drag.originalCenter, delta);

    // updateCenter(drag.componentIndex, drag.elementIndex, newCenter);
    scheduleSend(drag.eventName, {
      index: drag.elementIndex,
      position: newCenter,
    });
  };

  const handleDragEnd = () => {
    dragRef.current = null;
    window.removeEventListener("pointermove", handleDragMove);
  };

  if (!scene) {
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
        components={scene.components}
        defaultCamera={scene.defaultCamera || DEFAULT_CAMERA}
        controls={scene.controls}
        onCameraChange={setCamera}
        onCanvasRef={(canvas) => { canvasRef.current = canvas; }}
        pointerRef={pointerRef}
        cursor="auto"
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
