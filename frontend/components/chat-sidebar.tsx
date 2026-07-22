"use client";

import { PanelLeftIcon, PlusIcon, Trash2Icon } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import type { SessionSummary } from "@/lib/sessions";

function relativeTime(iso: string): string {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const diff = Date.now() - then;
  const mins = Math.round(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.round(hrs / 24);
  return `${days}d ago`;
}

interface ChatSidebarProps {
  sessions: SessionSummary[];
  activeId: string | null;
  open: boolean;
  onToggle: () => void;
  onNewChat: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

export function ChatSidebar({
  sessions,
  activeId,
  open,
  onToggle,
  onNewChat,
  onSelect,
  onDelete,
}: ChatSidebarProps) {
  if (!open) {
    return (
      <div className="flex h-dvh shrink-0 flex-col border-r bg-background p-2">
        <Button
          variant="ghost"
          size="icon"
          aria-label="Open sidebar"
          onClick={onToggle}
        >
          <PanelLeftIcon />
        </Button>
      </div>
    );
  }

  return (
    <aside className="flex h-dvh w-64 shrink-0 flex-col border-r bg-background">
      <div className="flex items-center gap-2 p-3">
        <Button
          variant="ghost"
          size="icon"
          aria-label="Collapse sidebar"
          onClick={onToggle}
        >
          <PanelLeftIcon />
        </Button>
        <Button className="flex-1 justify-start" onClick={onNewChat}>
          <PlusIcon className="mr-2" />
          New chat
        </Button>
      </div>
      <Separator />
      <div className="min-h-0 w-full flex-1 overflow-y-auto">
        <div className="flex w-full flex-col gap-1 p-2">
          {sessions.length === 0 ? (
            <p className="px-2 py-4 text-center text-sm text-muted-foreground">
              No chats yet
            </p>
          ) : (
            sessions.map((s) => (
              <div
                key={s.id}
                className={cn(
                  "group/row flex w-full items-center gap-1 overflow-hidden rounded-lg",
                  s.id === activeId && "bg-muted"
                )}
              >
                <button
                  type="button"
                  onClick={() => onSelect(s.id)}
                  className="flex min-w-0 flex-1 flex-col items-start gap-0.5 overflow-hidden rounded-lg px-3 py-2 text-left hover:bg-muted"
                >
                  <span className="block w-full truncate text-sm font-medium">
                    {s.title}
                  </span>
                  {s.preview ? (
                    <span className="block w-full truncate text-xs text-muted-foreground">
                      {s.preview}
                    </span>
                  ) : null}
                  {s.updated_at ? (
                    <span className="text-[10px] text-muted-foreground">
                      {relativeTime(s.updated_at)}
                    </span>
                  ) : null}
                </button>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Delete chat"
                  className="mr-1 shrink-0 text-muted-foreground hover:text-destructive"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(s.id);
                  }}
                >
                  <Trash2Icon />
                </Button>
              </div>
            ))
          )}
        </div>
      </div>
    </aside>
  );
}
