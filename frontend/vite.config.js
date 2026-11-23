import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the page calls /api on the Vite server, which forwards it to the FastAPI container.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    watch: { usePolling: true }, // file changes on a Windows folder mounted into Docker
    proxy: {
      "/api": { target: process.env.API_URL || "http://localhost:8000", changeOrigin: true },
    },
  },
});
