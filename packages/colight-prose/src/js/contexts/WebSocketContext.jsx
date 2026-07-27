import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  useCallback,
} from "react";
import { WebsocketBuilder, ExponentialBackoff, ArrayQueue } from "websocket-ts";
import createLogger from "../logger.js";

const logger = createLogger("websocket-context");

const WebSocketContext = createContext(null);

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error("useWebSocket must be used within a WebSocketProvider");
  }
  return context;
};

export const WebSocketProvider = ({ children }) => {
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);
  const handlersRef = useRef(new Map()); // Map of handler id -> handler config
  const handlerIdCounter = useRef(0);

  // Register a message handler
  const registerHandler = useCallback((config) => {
    const id = ++handlerIdCounter.current;
    handlersRef.current.set(id, {
      handler: config.handler,
      types: config.types || null, // null means handle all types
      priority: config.priority || 0,
    });

    logger.debug(`Registered handler ${id} for types:`, config.types);

    // Return unregister function
    return () => {
      handlersRef.current.delete(id);
      logger.debug(`Unregistered handler ${id}`);
    };
  }, []);

  // Process incoming messages through all registered handlers
  const processMessage = useCallback((data) => {
    // Sort handlers by priority (higher priority first)
    const handlers = Array.from(handlersRef.current.entries()).sort(
      ([, a], [, b]) => b.priority - a.priority,
    );

    // Execute handlers
    for (const [id, config] of handlers) {
      // Check if handler wants this message type
      if (config.types && !config.types.includes(data.type)) {
        continue;
      }

      try {
        config.handler(data);
      } catch (error) {
        logger.error(`Handler ${id} threw error:`, error);
      }
    }
  }, []);

  useEffect(() => {
    const wsPort = parseInt(window.location.port) + 1;

    // Build WebSocket with exponential backoff and message buffering
    const ws = new WebsocketBuilder(`ws://127.0.0.1:${wsPort}`)
      .withBackoff(new ExponentialBackoff(1000, 2, 30000)) // 1s initial, 2x multiplier, 30s max
      .withBuffer(new ArrayQueue()) // Buffer messages when disconnected
      .onOpen(() => {
        logger.info("LiveServer connected");
        setConnected(true);
      })
      .onClose(() => {
        logger.info("LiveServer disconnected");
        setConnected(false);
      })
      .onError((_, error) => {
        logger.error("WebSocket error:", error);
      })
      .onMessage((_, event) => {
        const data = JSON.parse(event.data);
        logger.debug("WebSocket message:", data);
        processMessage(data);
      })
      .onRetry(() => {
        logger.info("WebSocket reconnecting...");
      })
      .build();

    wsRef.current = ws;

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [processMessage]);

  const sendMessage = useCallback((message) => {
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  const value = {
    connected,
    sendMessage,
    registerHandler,
    ws: wsRef.current,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};

// Hook for components to register message handlers
export const useMessageHandler = (config) => {
  const { registerHandler } = useWebSocket();
  const configRef = useRef();

  // Update config ref when it changes
  useEffect(() => {
    configRef.current = config;
  });

  useEffect(() => {
    if (!config.handler) return;

    // Create a stable handler that uses the ref
    const stableConfig = {
      handler: (...args) => configRef.current.handler(...args),
      types: config.types,
      priority: config.priority,
    };

    const unregister = registerHandler(stableConfig);
    return unregister;
  }, [config.types, config.priority, registerHandler]);
};
