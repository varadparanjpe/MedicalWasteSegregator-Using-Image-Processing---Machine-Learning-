import { useState, useEffect, useCallback } from "react";

export type Status = "idle" | "loading" | "success" | "error";

export function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("idle");

  const run = useCallback(async () => {
    setStatus("loading");
    setError(null);
    try {
      const result = await fn();
      setData(result);
      setStatus("success");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => { run(); }, [run]);

  return { data, error, status, reload: run, loading: status === "loading" };
}

export function usePoll<T>(fn: () => Promise<T>, intervalMs: number, deps: unknown[] = []) {
  const result = useAsync(fn, deps);
  useEffect(() => {
    const t = setInterval(result.reload, intervalMs);
    return () => clearInterval(t);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs]);
  return result;
}
