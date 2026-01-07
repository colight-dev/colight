import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  root: "web",
  resolve: {
    alias: {
      "@colight/scene3d": path.resolve(
        __dirname,
        "../../packages/colight/src/js/scene3d"
      ),
      "@colight/serde": path.resolve(
        __dirname,
        "../../packages/colight-serde/src/js"
      ),
    },
  },
  server: {
    port: 8000,
    strictPort: true,
  },
  build: {
    outDir: "../dist",
    emptyOutDir: true,
  },
});
