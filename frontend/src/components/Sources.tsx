import type { SourceInfo } from "../api/rag";

export function Sources({ sources }: { sources: SourceInfo[] }) {
  if (!sources?.length) return null;
  return (
    <div className="mt-2 flex flex-wrap gap-2 text-xs text-zinc-500">
      <span className="text-zinc-400">Sources:</span>
      {sources.map((s, idx) => (
        <span
          key={`${s.source}-${s.source_type}-${idx}`}
          className="rounded-full border border-zinc-800 bg-zinc-950/40 px-2 py-1"
          title={s.source}
        >
          {s.source} ({s.source_type})
        </span>
      ))}
    </div>
  );
}

