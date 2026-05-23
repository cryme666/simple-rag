import { createClient, createConfig } from "./generated/client";

const baseUrl = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export const apiClient = createClient(
  createConfig({
    baseUrl,
  }),
);

