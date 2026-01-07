import React, {
  useEffect,
  useMemo,
  useRef,
  useState,
  useCallback,
  createContext,
  useContext,
} from "react";
import { createRoot } from "react-dom/client";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, Box } from "@react-three/drei";
import * as THREE from "three";
import { packMessage, unpackMessage } from "../../../packages/colight-wire-protocol/dist/index.mjs";

const WS_URL = `ws://${window.location.hostname}:8001`;

const DEFAULT_CAMERA = {
  position: [2.2, 2.2, 1.8],
  target: [0, 0, 0],
  up: [0, 0, 1],
  fov: 45,
  near: 0.01,
  far: 100,
};

// Context for sharing drag state across components
const DragContext = createContext(null);

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

  const sendEvent = useCallback((eventName, data) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const [envelope, buffers] = packMessage({
      type: "event",
      event: eventName,
      data,
    });
    ws.send(JSON.stringify(envelope));
    buffers.forEach((buffer) => ws.send(buffer));
  }, []);

  return { scene, connected, sendEvent };
}

// Draggable Cuboid component
function DraggableCuboid({
  center,
  halfSize,
  color,
  alpha,
  index,
  dragEventName,
  onDrag,
}) {
  const meshRef = useRef();
  const [hovered, setHovered] = useState(false);
  const { camera, gl } = useThree();
  const dragContext = useContext(DragContext);

  // Convert color array to THREE.Color
  const threeColor = useMemo(() => {
    if (Array.isArray(color)) {
      return new THREE.Color(color[0], color[1], color[2]);
    }
    return new THREE.Color(color);
  }, [color]);

  // Handle pointer down - start drag
  const handlePointerDown = useCallback(
    (event) => {
      if (!dragEventName) return;

      event.stopPropagation();

      // Get the face normal in world space
      const face = event.face;
      if (!face) return;

      const normal = face.normal.clone();
      // Transform normal to world space
      normal.transformDirection(meshRef.current.matrixWorld);

      // Create drag plane at hit point with face normal
      const dragPlane = new THREE.Plane();
      dragPlane.setFromNormalAndCoplanarPoint(normal, event.point);

      // Calculate offset from hit point to center
      const centerVec = new THREE.Vector3(center[0], center[1], center[2]);
      const dragOffset = new THREE.Vector3().subVectors(centerVec, event.point);

      // Store drag state in context
      dragContext.startDrag({
        index,
        dragEventName,
        dragPlane,
        dragOffset,
        onDrag,
      });

      gl.domElement.style.cursor = "grabbing";
    },
    [dragEventName, center, index, onDrag, dragContext, gl],
  );

  // Handle hover
  const handlePointerOver = useCallback(() => {
    if (dragEventName && !dragContext.isDragging) {
      setHovered(true);
      gl.domElement.style.cursor = "grab";
    }
  }, [dragEventName, dragContext.isDragging, gl]);

  const handlePointerOut = useCallback(() => {
    setHovered(false);
    if (!dragContext.isDragging) {
      gl.domElement.style.cursor = "auto";
    }
  }, [dragContext.isDragging, gl]);

  // Half size handling
  const size = useMemo(() => {
    if (typeof halfSize === "number") {
      return [halfSize * 2, halfSize * 2, halfSize * 2];
    }
    return [halfSize[0] * 2, halfSize[1] * 2, halfSize[2] * 2];
  }, [halfSize]);

  return (
    <Box
      ref={meshRef}
      args={size}
      position={center}
      onPointerDown={handlePointerDown}
      onPointerOver={handlePointerOver}
      onPointerOut={handlePointerOut}
    >
      <meshStandardMaterial
        color={threeColor}
        transparent={alpha < 1}
        opacity={alpha}
      />
    </Box>
  );
}

// Component to handle drag movement via window events
function DragHandler() {
  const { camera, gl } = useThree();
  const dragContext = useContext(DragContext);
  const raycaster = useMemo(() => new THREE.Raycaster(), []);

  useEffect(() => {
    const handlePointerMove = (event) => {
      const dragState = dragContext.dragStateRef.current;
      if (!dragState) return;

      // Cast ray from mouse position
      const rect = gl.domElement.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1,
      );
      raycaster.setFromCamera(mouse, camera);

      // Find intersection with drag plane
      const intersection = new THREE.Vector3();
      if (raycaster.ray.intersectPlane(dragState.dragPlane, intersection)) {
        // Add offset to get new center position
        const newCenter = intersection.add(dragState.dragOffset);
        dragState.onDrag(dragState.index, [newCenter.x, newCenter.y, newCenter.z]);
      }
    };

    const handlePointerUp = () => {
      if (dragContext.dragStateRef.current) {
        dragContext.endDrag();
        gl.domElement.style.cursor = "auto";
      }
    };

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);

    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, [camera, gl, dragContext, raycaster]);

  return null;
}

// Scene content component (inside Canvas)
function SceneContent({ sceneData, onDrag, isDragging }) {
  const { camera } = useThree();
  const controlsRef = useRef();

  // Set up camera from scene data
  useEffect(() => {
    if (sceneData?.defaultCamera) {
      const cam = sceneData.defaultCamera;
      camera.position.set(cam.position[0], cam.position[1], cam.position[2]);
      camera.up.set(cam.up[0], cam.up[1], cam.up[2]);
      camera.fov = cam.fov;
      camera.near = cam.near;
      camera.far = cam.far;
      camera.updateProjectionMatrix();

      if (controlsRef.current) {
        controlsRef.current.target.set(
          cam.target[0],
          cam.target[1],
          cam.target[2],
        );
        controlsRef.current.update();
      }
    }
  }, [sceneData?.defaultCamera, camera]);

  if (!sceneData) return null;

  return (
    <>
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 10]} intensity={1} />
      <directionalLight position={[-10, -10, -10]} intensity={0.3} />

      <OrbitControls
        ref={controlsRef}
        enableDamping={false}
        enabled={!isDragging}
      />

      <DragHandler />

      {sceneData.components.map((component, idx) => {
        if (component.type === "Cuboid") {
          const centers = component.centers;
          const numCuboids = centers.length / 3;
          const dragToken = component.drag?.__event__;

          return Array.from({ length: numCuboids }, (_, i) => {
            const center = [
              centers[i * 3],
              centers[i * 3 + 1],
              centers[i * 3 + 2],
            ];

            return (
              <DraggableCuboid
                key={`${idx}-${i}`}
                center={center}
                halfSize={component.half_size}
                color={component.color}
                alpha={component.alpha ?? 1}
                index={i}
                dragEventName={dragToken}
                onDrag={(index, position) =>
                  onDrag(dragToken, { index, position })
                }
              />
            );
          });
        }
        return null;
      })}

      {/* Grid helper for reference */}
      <gridHelper args={[10, 10]} rotation={[Math.PI / 2, 0, 0]} />
    </>
  );
}

function App() {
  const { scene, connected, sendEvent } = useWireScene(WS_URL);
  const [localScene, setLocalScene] = useState(scene);
  const [isDragging, setIsDragging] = useState(false);
  const sendRef = useRef(sendEvent);
  const rafRef = useRef(null);
  const pendingPayloadRef = useRef(null);
  const dragStateRef = useRef(null);

  useEffect(() => {
    setLocalScene(scene);
  }, [scene]);

  useEffect(() => {
    sendRef.current = sendEvent;
  }, [sendEvent]);

  const scheduleSend = useCallback((eventName, payload) => {
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
  }, []);

  const handleDrag = useCallback(
    (eventName, { index, position }) => {
      // Update local state immediately for responsiveness
      setLocalScene((prev) => {
        if (!prev) return prev;
        const components = [...prev.components];
        const component = components.find(
          (c) => c.drag?.__event__ === eventName,
        );
        if (!component?.centers) return prev;

        const centers = component.centers;
        const nextCenters =
          centers instanceof Float32Array
            ? centers.slice()
            : Float32Array.from(centers);
        const base = index * 3;
        nextCenters[base] = position[0];
        nextCenters[base + 1] = position[1];
        nextCenters[base + 2] = position[2];

        const componentIndex = components.indexOf(component);
        components[componentIndex] = { ...component, centers: nextCenters };
        return { ...prev, components };
      });

      // Send to server (throttled)
      scheduleSend(eventName, { index, position });
    },
    [scheduleSend],
  );

  const startDrag = useCallback((state) => {
    dragStateRef.current = state;
    setIsDragging(true);
  }, []);

  const endDrag = useCallback(() => {
    dragStateRef.current = null;
    setIsDragging(false);
  }, []);

  const dragContextValue = useMemo(
    () => ({
      isDragging,
      dragStateRef,
      startDrag,
      endDrag,
    }),
    [isDragging, startDrag, endDrag],
  );

  if (!localScene) {
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

  const cam = localScene.defaultCamera || DEFAULT_CAMERA;

  return (
    <DragContext.Provider value={dragContextValue}>
      <div style={{ width: "100%", height: "100%", position: "relative" }}>
        <Canvas
          camera={{
            position: cam.position,
            up: cam.up,
            fov: cam.fov,
            near: cam.near,
            far: cam.far,
          }}
        >
          <SceneContent
            sceneData={localScene}
            onDrag={handleDrag}
            isDragging={isDragging}
          />
        </Canvas>
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
          {connected ? "Drag the cube (R3F)" : "Connecting..."}
        </div>
      </div>
    </DragContext.Provider>
  );
}

const root = createRoot(document.getElementById("root"));
root.render(<App />);
