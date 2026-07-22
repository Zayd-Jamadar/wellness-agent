"use client";

import { useChat } from "@ai-sdk/react";
import {
  DefaultChatTransport,
  type ToolUIPart,
  type UIMessage,
} from "ai";
import { useCallback, useEffect, useMemo, useState } from "react";

import { ChatSidebar } from "@/components/chat-sidebar";
import {
  createSession,
  deleteSession,
  getSession,
  listSessions,
  type SessionSummary,
} from "@/lib/sessions";
import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputBody,
  type PromptInputMessage,
  PromptInputSubmit,
  PromptInputTextarea,
} from "@/components/ai-elements/prompt-input";
import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger,
} from "@/components/ai-elements/reasoning";
import {
  Tool,
  ToolContent,
  ToolHeader,
  ToolInput,
  ToolOutput,
} from "@/components/ai-elements/tool";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function Page() {
  // Conversation id is minted by the backend (LangGraph-style thread create).
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [initialMessages, setInitialMessages] = useState<UIMessage[]>([]);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const refreshSessions = useCallback(() => {
    listSessions().then(setSessions).catch(() => {});
  }, []);

  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

  useEffect(() => {
    createSession().then(setConversationId);
  }, []);

  const handleNewChat = useCallback(async () => {
    const id = await createSession();
    setInitialMessages([]);
    setConversationId(id);
  }, []);

  const handleSelect = useCallback(async (id: string) => {
    const history = await getSession(id);
    setInitialMessages(history);
    setConversationId(id);
  }, []);

  const handleDelete = useCallback(
    async (id: string) => {
      await deleteSession(id);
      if (id === conversationId) {
        const fresh = await createSession();
        setInitialMessages([]);
        setConversationId(fresh);
      }
      refreshSessions();
    },
    [conversationId, refreshSessions]
  );

  return (
    <div className="flex h-dvh w-full">
      <ChatSidebar
        sessions={sessions}
        activeId={conversationId}
        open={sidebarOpen}
        onToggle={() => setSidebarOpen((v) => !v)}
        onNewChat={handleNewChat}
        onSelect={handleSelect}
        onDelete={handleDelete}
      />
      {/* Remount the chat whenever the conversation changes so useChat gets a
          fresh Chat instance bound to the new thread id (SDK v5 does not
          reliably reset on id changes for an existing instance). */}
      {conversationId ? (
        <Chat
          key={conversationId}
          conversationId={conversationId}
          initialMessages={initialMessages}
          onFinish={refreshSessions}
        />
      ) : (
        <main className="flex-1" />
      )}
    </div>
  );
}

interface ChatProps {
  conversationId: string;
  initialMessages: UIMessage[];
  onFinish: () => void;
}

function Chat({ conversationId, initialMessages, onFinish }: ChatProps) {
  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: `${API_URL}/api/chat`,
        prepareSendMessagesRequest: ({ messages }) => ({
          body: { id: conversationId, messages },
        }),
      }),
    [conversationId]
  );

  const { messages, sendMessage, status } = useChat({
    id: conversationId,
    messages: initialMessages,
    transport,
    onFinish,
  });

  const handleSubmit = (message: PromptInputMessage) => {
    if (!message.text?.trim()) return;
    sendMessage({ text: message.text });
  };

  return (
    <main className="flex min-w-0 flex-1 flex-col">
      <Conversation className="min-h-0 flex-1">
        <ConversationContent className="mx-auto w-full max-w-3xl">
          {messages.length === 0 ? (
            <ConversationEmptyState
              title="Wellness Assistant"
              description="Ask about diet, exercise, sleep, or supplements."
            />
          ) : (
            messages.map((message) => (
              <Message from={message.role} key={message.id}>
                <MessageContent>
                  {message.parts.map((part, i) => {
                    const key = `${message.id}-${i}`;

                    if (part.type === "text") {
                      return (
                        <MessageResponse key={key}>{part.text}</MessageResponse>
                      );
                    }

                    if (part.type === "reasoning") {
                      return (
                        <Reasoning
                          key={key}
                          className="w-full"
                          isStreaming={status === "streaming"}
                        >
                          <ReasoningTrigger />
                          <ReasoningContent>{part.text}</ReasoningContent>
                        </Reasoning>
                      );
                    }

                    if (part.type.startsWith("tool-")) {
                      const tool = part as ToolUIPart;
                      return (
                        <Tool key={key}>
                          <ToolHeader type={tool.type} state={tool.state} />
                          <ToolContent>
                            <ToolInput input={tool.input} />
                            <ToolOutput
                              output={tool.output}
                              errorText={tool.errorText}
                            />
                          </ToolContent>
                        </Tool>
                      );
                    }

                    return null;
                  })}
                </MessageContent>
              </Message>
            ))
          )}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>

      <div className="mx-auto w-full max-w-3xl p-4">
        <PromptInput onSubmit={handleSubmit}>
          <PromptInputBody>
            <PromptInputTextarea placeholder="Ask about diet, exercise, sleep…" />
          </PromptInputBody>
          <PromptInputSubmit status={status} />
        </PromptInput>
      </div>
    </main>
  );
}
