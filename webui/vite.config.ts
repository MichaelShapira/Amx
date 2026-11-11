import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Local application: the dev server listens on loopback only and proxies /api to the Python
// API, which also binds 127.0.0.1. Nothing is exposed beyond this machine.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 8600,
    strictPort: true,
    proxy: { "/api": { target: "http://127.0.0.1:8601", changeOrigin: false } },
  },
  preview: { host: "127.0.0.1", port: 8600 },
  build: { outDir: "dist", sourcemap: false },
});
