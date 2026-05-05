import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  server: {
    port: 5173,
    open: true,
  },
  build: {
    target: "es2020",
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          "react-vendor":   ["react", "react-dom", "react-router-dom"],
          "recharts":       ["recharts"],
          "webcam":         ["react-webcam"],
          "dropzone":       ["react-dropzone"],
          "icons":          ["lucide-react"],
        },
      },
    },
    chunkSizeWarningLimit: 600,
  },
});
