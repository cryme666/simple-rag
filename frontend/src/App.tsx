import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { chat, checkHealth, clearKnowledgeBase, ingestPdf, ingestUrl, type ChatMessage, type SourceInfo } from "./api/rag";
import { Sources } from "./components/Sources";
import { TypingDots } from "./components/TypingDots";

type UiMessage =
  | { id: string; role: "user"; content: string }
  | { id: string; role: "assistant"; content: string; sources?: SourceInfo[] };

function toConversationHistory(messages: UiMessage[]): ChatMessage[] {
  return messages
    .filter((m) => m.role === "user" || m.role === "assistant")
    .map((m) => ({ role: m.role, content: m.content }));
}

function App() {
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [pdfOverwrite, setPdfOverwrite] = useState(false);
  const [pdfStatus, setPdfStatus] = useState<{ kind: "idle" | "ok" | "err"; text: string }>({
    kind: "idle",
    text: "",
  });

  const [urlValue, setUrlValue] = useState("");
  const [urlOverwrite, setUrlOverwrite] = useState(false);
  const [urlStatus, setUrlStatus] = useState<{ kind: "idle" | "ok" | "err"; text: string }>({
    kind: "idle",
    text: "",
  });

  const [kbStatus, setKbStatus] = useState<{ kind: "idle" | "ok" | "err"; text: string }>({
    kind: "idle",
    text: "",
  });

  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [healthOk, setHealthOk] = useState<boolean | null>(null);

  const scrollerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function pollHealth() {
      try {
        const res = await checkHealth();
        if (!cancelled) setHealthOk(res.status === "ok");
      } catch {
        if (!cancelled) setHealthOk(false);
      }
    }

    pollHealth();
    const id = window.setInterval(pollHealth, 15000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  useEffect(() => {
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, isSending]);

  const conversationHistory = useMemo(() => toConversationHistory(messages), [messages]);

  async function onUploadPdf(e: React.FormEvent) {
    e.preventDefault();
    if (!pdfFile) return;
    setPdfStatus({ kind: "idle", text: "" });
    try {
      const res = await ingestPdf(pdfFile, pdfOverwrite);
      setPdfStatus({ kind: "ok", text: `${res.message} Chunks: ${res.chunks_stored}` });
    } catch (err) {
      setPdfStatus({ kind: "err", text: err instanceof Error ? err.message : "Upload failed" });
    }
  }

  async function onIngestUrl(e: React.FormEvent) {
    e.preventDefault();
    const url = urlValue.trim();
    if (!url) return;
    setUrlStatus({ kind: "idle", text: "" });
    try {
      const res = await ingestUrl(url, urlOverwrite);
      setUrlStatus({
        kind: "ok",
        text: `${res.message} Chunks: ${res.chunks_stored} (${res.title})`,
      });
    } catch (err) {
      setUrlStatus({ kind: "err", text: err instanceof Error ? err.message : "Ingest failed" });
    }
  }

  async function onClearKb() {
    setKbStatus({ kind: "idle", text: "" });
    try {
      const res = await clearKnowledgeBase();
      setKbStatus({ kind: "ok", text: `${res.message} Deleted: ${res.deleted_count}` });
    } catch (err) {
      setKbStatus({ kind: "err", text: err instanceof Error ? err.message : "Clear failed" });
    }
  }

  async function onSend(e: React.FormEvent) {
    e.preventDefault();
    const msg = input.trim();
    if (!msg || isSending) return;

    const userMsg: UiMessage = { id: crypto.randomUUID(), role: "user", content: msg };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setIsSending(true);

    try {
      const res = await chat(msg, [...conversationHistory, { role: "user", content: msg }]);
      const assistantMsg: UiMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: res.response,
        sources: (res.sources ?? []) as SourceInfo[],
      };
      setMessages((m) => [...m, assistantMsg]);
    } catch (err) {
      const assistantMsg: UiMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: err instanceof Error ? `Error: ${err.message}` : "Error: request failed",
      };
      setMessages((m) => [...m, assistantMsg]);
    } finally {
      setIsSending(false);
    }
  }

  function onClearChat() {
    setMessages([]);
  }

  return (
    <div className="h-full bg-zinc-950 text-zinc-100">
      <div className="mx-auto grid h-full max-w-6xl grid-cols-1 gap-4 p-4 lg:grid-cols-[360px_1fr]">
        <aside className="rounded-2xl border border-zinc-900 bg-zinc-950/40 p-4">
          <div className="mb-4">
            <div className="text-lg font-semibold">Knowledge base</div>
            <div className="text-sm text-zinc-400">Ingest PDFs and URLs for retrieval.</div>
          </div>

          <form onSubmit={onUploadPdf} className="mb-6 space-y-3">
            <div className="text-sm font-medium">Upload PDF</div>
            <input
              type="file"
              accept="application/pdf"
              className="block w-full text-sm text-zinc-300 file:mr-3 file:rounded-lg file:border file:border-zinc-800 file:bg-zinc-900 file:px-3 file:py-2 file:text-sm file:text-zinc-100"
              onChange={(e) => setPdfFile(e.target.files?.[0] ?? null)}
            />
            <label className="flex items-center gap-2 text-sm text-zinc-300">
              <input
                type="checkbox"
                className="h-4 w-4 rounded border-zinc-700 bg-zinc-900"
                checked={pdfOverwrite}
                onChange={(e) => setPdfOverwrite(e.target.checked)}
              />
              Overwrite if already ingested
            </label>
            <button
              type="submit"
              disabled={!pdfFile}
              className="w-full rounded-xl bg-indigo-600 px-4 py-2 text-sm font-semibold disabled:opacity-50"
            >
              Ingest PDF
            </button>
            {pdfStatus.text ? (
              <div className={`text-sm ${pdfStatus.kind === "err" ? "text-rose-400" : "text-emerald-400"}`}>
                {pdfStatus.text}
              </div>
            ) : null}
          </form>

          <form onSubmit={onIngestUrl} className="mb-6 space-y-3">
            <div className="text-sm font-medium">Ingest URL</div>
            <input
              value={urlValue}
              onChange={(e) => setUrlValue(e.target.value)}
              placeholder="https://example.com/article"
              className="w-full rounded-xl border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm outline-none focus:border-indigo-500"
            />
            <label className="flex items-center gap-2 text-sm text-zinc-300">
              <input
                type="checkbox"
                className="h-4 w-4 rounded border-zinc-700 bg-zinc-900"
                checked={urlOverwrite}
                onChange={(e) => setUrlOverwrite(e.target.checked)}
              />
              Overwrite if already ingested
            </label>
            <button
              type="submit"
              disabled={!urlValue.trim()}
              className="w-full rounded-xl bg-indigo-600 px-4 py-2 text-sm font-semibold disabled:opacity-50"
            >
              Ingest URL
            </button>
            {urlStatus.text ? (
              <div className={`text-sm ${urlStatus.kind === "err" ? "text-rose-400" : "text-emerald-400"}`}>
                {urlStatus.text}
              </div>
            ) : null}
          </form>

          <div className="space-y-3">
            <div className="text-sm font-medium">Maintenance</div>
            <button
              type="button"
              onClick={onClearKb}
              className="w-full rounded-xl border border-zinc-800 bg-zinc-950 px-4 py-2 text-sm font-semibold"
            >
              Clear knowledge base
            </button>
            {kbStatus.text ? (
              <div className={`text-sm ${kbStatus.kind === "err" ? "text-rose-400" : "text-emerald-400"}`}>
                {kbStatus.text}
              </div>
            ) : null}
          </div>
        </aside>

        <main className="flex min-h-0 flex-col rounded-2xl border border-zinc-900 bg-zinc-950/40">
          <div className="flex items-center justify-between border-b border-zinc-900 px-4 py-3">
            <div>
              <div className="flex items-center gap-2">
                <div className="text-lg font-semibold">RAG chat</div>
                <span
                  className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs ${
                    healthOk === null
                      ? "bg-zinc-800 text-zinc-400"
                      : healthOk
                        ? "bg-emerald-950 text-emerald-400"
                        : "bg-rose-950 text-rose-400"
                  }`}
                  title="Backend health"
                >
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${
                      healthOk === null ? "bg-zinc-500" : healthOk ? "bg-emerald-400" : "bg-rose-400"
                    }`}
                  />
                  {healthOk === null ? "Checking…" : healthOk ? "Online" : "Offline"}
                </span>
              </div>
              <div className="text-xs text-zinc-500">Model: Groq llama-3.3-70b-versatile</div>
            </div>
            <button
              type="button"
              onClick={onClearChat}
              className="rounded-xl border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm"
            >
              Clear chat
            </button>
          </div>

          <div ref={scrollerRef} className="min-h-0 flex-1 space-y-3 overflow-auto p-4">
            {messages.length === 0 ? (
              <div className="rounded-xl border border-zinc-900 bg-zinc-950/60 p-4 text-sm text-zinc-400">
                Ingest a PDF/URL, then ask a question. Sources will appear under the answer.
              </div>
            ) : null}
            {messages.map((m) => (
              <div
                key={m.id}
                className={
                  m.role === "user"
                    ? "ml-auto max-w-[85%] rounded-2xl bg-indigo-600 px-4 py-3 text-sm text-white"
                    : "mr-auto max-w-[85%] rounded-2xl border border-zinc-900 bg-zinc-950/60 px-4 py-3 text-sm"
                }
              >
                {m.role === "assistant" ? (
                  <div className="prose prose-invert prose-sm max-w-none">
                    <ReactMarkdown>{m.content}</ReactMarkdown>
                  </div>
                ) : (
                  m.content
                )}
                {"sources" in m && m.sources ? <Sources sources={m.sources} /> : null}
              </div>
            ))}

            {isSending ? (
              <div className="mr-auto max-w-[85%] rounded-2xl border border-zinc-900 bg-zinc-950/60 px-4 py-3 text-sm">
                <TypingDots />
              </div>
            ) : null}
          </div>

          <form onSubmit={onSend} className="flex items-center gap-2 border-t border-zinc-900 p-3">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question…"
              className="flex-1 rounded-xl border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm outline-none focus:border-indigo-500"
            />
            <button
              type="submit"
              disabled={!input.trim() || isSending}
              className="rounded-xl bg-indigo-600 px-4 py-2 text-sm font-semibold disabled:opacity-50"
            >
              Send
            </button>
          </form>
        </main>
      </div>
    </div>
  );
}

export default App;
