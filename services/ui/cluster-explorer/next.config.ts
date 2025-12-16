import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable standalone output for Docker deployment
  output: "standalone",

  // Environment variables for API
  env: {
    API_URL: process.env.API_URL || "http://localhost:8000",
  },
};

export default nextConfig;
