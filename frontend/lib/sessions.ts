import type { UIMessage } from "ai";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface SessionSummary {
  id: string;
  title: string;
  preview: string;
  updated_at: string;
}

interface SessionDetailResponse {
  id: string;
  title: string;
  messages: Array<{
    id?: string | null;
    role: string;
    parts: Array<{ type: string; text?: string | null }>;
  }>;
}

export async function createSession(): Promise<string> {
  const res = await fetch(`${API_URL}/api/sessions`, { method: "POST" });
  if (!res.ok) return crypto.randomUUID();
  const data = (await res.json()) as { id: string };
  return data.id;
}

export async function listSessions(): Promise<SessionSummary[]> {
  const res = await fetch(`${API_URL}/api/sessions`);
  if (!res.ok) return [];
  return (await res.json()) as SessionSummary[];
}

export async function getSession(id: string): Promise<UIMessage[]> {
  const res = await fetch(`${API_URL}/api/sessions/${id}`);
  if (!res.ok) return [];
  const detail = (await res.json()) as SessionDetailResponse;
  return detail.messages.map((m, i) => ({
    id: m.id ?? `${id}-${i}`,
    role: m.role as UIMessage["role"],
    parts: m.parts.map((p) => ({
      type: p.type as "text",
      text: p.text ?? "",
    })),
  })) as UIMessage[];
}

export async function deleteSession(id: string): Promise<void> {
  await fetch(`${API_URL}/api/sessions/${id}`, { method: "DELETE" });
}
