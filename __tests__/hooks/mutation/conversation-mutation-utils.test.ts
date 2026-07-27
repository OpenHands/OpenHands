/**
 * Tests for patchConversationInCache and updateConversationExecutionStatusInCache
 * in conversation-mutation-utils.ts.
 *
 * These functions are the mechanism that keeps the React Query cache in sync
 * with optimistic UI updates after stop/resume mutations. The most critical
 * invariant is that after pauseConversation succeeds the cache carries BOTH
 * execution_status AND sandbox_status as "PAUSED" — if sandbox_status is not
 * updated, WebSocketProviderWrapper's sandbox_status === "PAUSED" gate never
 * fires and the WebSocket tries to connect to the dead sandbox.
 */

import { QueryClient } from "@tanstack/react-query";
import { describe, expect, it, beforeEach } from "vitest";
import {
  patchConversationInCache,
  updateConversationExecutionStatusInCache,
} from "#/hooks/mutation/conversation-mutation-utils";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";

const CONV_ID = "conv-test-1";

const makeConversation = (
  overrides: Partial<AppConversation> = {},
): AppConversation => ({
  id: CONV_ID,
  created_by_user_id: null,
  selected_repository: null,
  selected_branch: null,
  git_provider: null,
  title: "Test conversation",
  trigger: null,
  pr_number: [],
  llm_model: null,
  metrics: null,
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
  execution_status: null,
  conversation_url: "https://sandbox.example.com/api/conversations/conv-test-1",
  session_api_key: "sess-key",
  sandbox_id: "sbx-1",
  sandbox_status: "RUNNING",
  sub_conversation_ids: [],
  ...overrides,
});

let queryClient: QueryClient;

beforeEach(() => {
  queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
});

// ─── patchConversationInCache ─────────────────────────────────────────────────

describe("patchConversationInCache", () => {
  it("patches the single-item conversation cache", () => {
    queryClient.setQueryData(
      ["user", "conversation", CONV_ID],
      makeConversation({ sandbox_status: "RUNNING" }),
    );

    patchConversationInCache(queryClient, CONV_ID, { sandbox_status: "PAUSED" });

    const cached = queryClient.getQueryData<AppConversation>([
      "user",
      "conversation",
      CONV_ID,
    ]);
    expect(cached?.sandbox_status).toBe("PAUSED");
  });

  it("patches conversation inside the paginated list cache", () => {
    queryClient.setQueryData(["user", "conversations"], {
      pages: [{ items: [makeConversation({ sandbox_status: "RUNNING" })] }],
      pageParams: [],
    });

    patchConversationInCache(queryClient, CONV_ID, { sandbox_status: "PAUSED" });

    const list = queryClient.getQueryData<{
      pages: Array<{ items: AppConversation[] }>;
    }>(["user", "conversations"]);
    expect(list?.pages[0]?.items[0]?.sandbox_status).toBe("PAUSED");
  });

  it("patches multiple fields atomically", () => {
    queryClient.setQueryData(
      ["user", "conversation", CONV_ID],
      makeConversation({
        execution_status: ExecutionStatus.RUNNING,
        sandbox_status: "RUNNING",
      }),
    );

    patchConversationInCache(queryClient, CONV_ID, {
      execution_status: ExecutionStatus.PAUSED,
      sandbox_status: "PAUSED",
    });

    const cached = queryClient.getQueryData<AppConversation>([
      "user",
      "conversation",
      CONV_ID,
    ]);
    expect(cached?.execution_status).toBe(ExecutionStatus.PAUSED);
    expect(cached?.sandbox_status).toBe("PAUSED");
  });

  it("does not modify other conversations in the list", () => {
    const other = makeConversation({
      id: "conv-other",
      sandbox_status: "RUNNING",
    });
    queryClient.setQueryData(["user", "conversations"], {
      pages: [{ items: [makeConversation(), other] }],
      pageParams: [],
    });

    patchConversationInCache(queryClient, CONV_ID, { sandbox_status: "PAUSED" });

    const list = queryClient.getQueryData<{
      pages: Array<{ items: AppConversation[] }>;
    }>(["user", "conversations"]);
    const otherCached = list?.pages[0]?.items.find((c) => c.id === "conv-other");
    expect(otherCached?.sandbox_status).toBe("RUNNING");
  });

  it("is a no-op when the single-item cache is empty", () => {
    // No query data seeded — should not throw.
    expect(() =>
      patchConversationInCache(queryClient, CONV_ID, { sandbox_status: "PAUSED" }),
    ).not.toThrow();
  });
});

// ─── updateConversationExecutionStatusInCache (backwards compat wrapper) ─────

describe("updateConversationExecutionStatusInCache", () => {
  it("updates execution_status without disturbing sandbox_status", () => {
    queryClient.setQueryData(
      ["user", "conversation", CONV_ID],
      makeConversation({
        execution_status: ExecutionStatus.RUNNING,
        sandbox_status: "RUNNING",
      }),
    );

    updateConversationExecutionStatusInCache(
      queryClient,
      CONV_ID,
      ExecutionStatus.PAUSED,
    );

    const cached = queryClient.getQueryData<AppConversation>([
      "user",
      "conversation",
      CONV_ID,
    ]);
    expect(cached?.execution_status).toBe(ExecutionStatus.PAUSED);
    // sandbox_status must remain unchanged — this wrapper only touches execution_status
    expect(cached?.sandbox_status).toBe("RUNNING");
  });
});
