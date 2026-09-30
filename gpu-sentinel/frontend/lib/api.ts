// Thin API client. The JWT lives in sessionStorage (cleared when the tab closes).
"use client";
import { useCallback, useEffect, useRef, useState } from "react";

const TOKEN_KEY = "sentinel.token";
const USER_KEY = "sentinel.user";

export type User = { username: string; role: "admin" | "operator" | "viewer"; tenant: string };

function store(): Storage | null {
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

export function getToken(): string | null {
  return store()?.getItem(TOKEN_KEY) ?? null;
}
export function getUser(): User | null {
  const raw = store()?.getItem(USER_KEY);
  return raw ? (JSON.parse(raw) as User) : null;
}
export function setSession(token: string, user: User) {
  store()?.setItem(TOKEN_KEY, token);
  store()?.setItem(USER_KEY, JSON.stringify(user));
}
export function clearSession() {
  store()?.removeItem(TOKEN_KEY);
  store()?.removeItem(USER_KEY);
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function api<T = any>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { "Content-Type": "application/json", ...(init.headers as any) };
  const t = getToken();
  if (t) headers.Authorization = `Bearer ${t}`;
  const r = await fetch(`/api/v1${path}`, { ...init, headers });
  if (r.status === 401 && !path.startsWith("/auth/login")) {
    clearSession();
    if (typeof window !== "undefined" && !window.location.pathname.startsWith("/login")) window.location.href = "/login/";
  }
  if (!r.ok) {
    let msg = r.statusText;
    try {
      const b = await r.json();
      msg = typeof b.detail === "string" ? b.detail : JSON.stringify(b.detail ?? b);
    } catch {}
    throw new ApiError(r.status, msg);
  }
  return r.json();
}

export const post = <T = any>(path: string, body?: unknown) =>
  api<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });
export const del = <T = any>(path: string) => api<T>(path, { method: "DELETE" });
export const patch = <T = any>(path: string, body: unknown) => api<T>(path, { method: "PATCH", body: JSON.stringify(body) });

/** Poll an endpoint. Returns data, error, loading and a manual reload. */
export function usePoll<T = any>(path: string | null, intervalMs = 10000) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const pathRef = useRef(path);
  pathRef.current = path;
  const load = useCallback(async () => {
    if (!pathRef.current) return;
    try {
      const d = await api<T>(pathRef.current);
      setData(d);
      setError(null);
    } catch (e: any) {
      setError(e.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    setLoading(true);
    load();
    if (!intervalMs) return;
    const id = setInterval(load, intervalMs);
    return () => clearInterval(id);
  }, [path, intervalMs, load]);
  return { data, error, loading, reload: load };
}

export function can(user: User | null, action: "operate" | "admin"): boolean {
  if (!user) return false;
  if (action === "admin") return user.role === "admin";
  return user.role === "admin" || user.role === "operator";
}
