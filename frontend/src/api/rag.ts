import { apiClient } from "./client";
import {
  chatChatPost,
  clearDatabaseIngestClearDelete,
  healthCheckHealthGet,
  ingestFileIngestFilePost,
  ingestUrlIngestUrlPost,
} from "./generated/sdk.gen";
import type { ChatResponse, IngestFileResponse, ScrapeResponse, SourceInfo } from "./generated/types.gen";

export type ChatRole = "user" | "assistant";

export type ChatMessage = {
  role: ChatRole;
  content: string;
};

export async function chat(message: string, conversationHistory: ChatMessage[]): Promise<ChatResponse> {
  const body = {
    message,
    conversation_history: conversationHistory,
  };
  const res = await chatChatPost({ client: apiClient, body, throwOnError: true });
  return res.data as ChatResponse;
}

export async function ingestPdf(file: File, overwrite: boolean): Promise<IngestFileResponse> {
  const res = await ingestFileIngestFilePost({
    client: apiClient,
    body: {
      file,
      overwrite,
    },
    throwOnError: true,
  });
  return res.data as IngestFileResponse;
}

export async function ingestUrl(url: string, overwrite: boolean): Promise<ScrapeResponse> {
  const res = await ingestUrlIngestUrlPost({
    client: apiClient,
    body: { url, overwrite },
    throwOnError: true,
  });
  return res.data as ScrapeResponse;
}

export async function clearKnowledgeBase(): Promise<{ message: string; deleted_count: number }> {
  const res = await clearDatabaseIngestClearDelete({ client: apiClient, throwOnError: true });
  return res.data as { message: string; deleted_count: number };
}

export async function checkHealth(): Promise<{ status: string }> {
  const res = await healthCheckHealthGet({ client: apiClient, throwOnError: true });
  return res.data as { status: string };
}

export type { SourceInfo };

