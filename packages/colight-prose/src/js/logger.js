// packages/colight-prose/src/js/logger.js

const LOG_LEVELS = {
  none: 0,
  error: 1,
  warn: 2,
  info: 3,
  debug: 4,
};

// Default to only showing warnings and errors.
// You can change this in the browser's developer console:
// localStorage.setItem('colight-log-level', 'debug'); // Show all logs
// localStorage.setItem('colight-log-level', 'info:CommandBar,warn:live'); // Component-specific levels
// localStorage.setItem('colight-log-level', 'warn'); // Go back to default
let config = {};

function parseConfig() {
  try {
    const configStr = localStorage.getItem("colight-log-level") || "warn";
    config = {};
    configStr.split(",").forEach((part) => {
      const [level, scope] = part.trim().split(":");
      if (scope) {
        config[scope] = level;
      } else {
        config["*"] = level; // Default level for all scopes
      }
    });
  } catch (e) {
    console.error("Failed to parse logging config:", e);
    config = { "*": "warn" };
  }
}

parseConfig();

// You can call this from the console to change log levels on the fly.
window.setColightLogLevel = (levelStr) => {
  localStorage.setItem("colight-log-level", levelStr);
  parseConfig();
  console.log(`Log level set. New config:`, config);
};

function createLogger(scope) {
  const getLogLevel = () => {
    const levelName = config[scope] || config["*"] || "warn";
    return LOG_LEVELS[levelName] || 0;
  };

  const log = (level, ...args) => {
    if (LOG_LEVELS[level] <= getLogLevel()) {
      // Use console.debug for debug level for better browser filtering
      const logFn = console[level] || console.log;
      logFn(`[${scope}]`, ...args);
    }
  };

  return {
    debug: (...args) => log("debug", ...args),
    info: (...args) => log("info", ...args),
    warn: (...args) => log("warn", ...args),
    error: (...args) => log("error", ...args),
  };
}

export default createLogger;
