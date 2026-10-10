import { http, HttpResponse } from "msw";
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useBtwInterceptor } from "#/hooks/chat/use-btw-interceptor";
import { server } from "#/mocks/node";
import { useBtwStore } from "#/stores/btw-store";

/**
 * Client contract for the `/btw` side-channel (see OpenHands/OpenHands
 * #17429: agent-server 500 `cannot pickle 'generator' object`).
 *
 * The failure lives server-side — this repo holds no `ask_agent`
 * implementation — so this test pins the two client-owned links instead,
 * end to end through the real SDK client (only the conversation lookup is
 * stubbed; it is a different service, out of scope here):
 *
 * 1. the wire shape the client sends (`POST .../ask_agent`, `{question}`);
 * 2. that a 500 carrying the server diagnostic envelope
 *    (`{detail, exception, error_id}`) reaches the btw error entry intact,
 *    preserving the `error_id` the server-side fix is diagnosed with.
 */
const CONV = "conv-1";
const QUESTION = "why?";
const AGENT_SERVER = "http://agent-server.test";

const SERVER_ERROR_BODY = {
  detail: "Internal Server Error",
  exception: "cannot pickle 'generator' object",
  error_id: "f6de8cc9a19847aea2c6b90ad4b5a4df",
};

const seenRequests: { method: string; pathname: string; body: unknown }[] = [];

const appConversation: AppConversation = {
  id: CONV,
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
  conversation_url: `${AGENT_SERVER}/api/conversations/${CONV}`,
  session_api_key: "session-key-1",
  sandbox_id: "sandbox-1",
  sandbox_status: "RUNNING",
  sub_conversation_ids: [],
};

const entries = () => useBtwStore.getState().entriesByConversation[CONV] ?? [];

describe("useBtwInterceptor agent-server error contract", () => {
  beforeEach(() => {
    useBtwStore.setState({ entriesByConversation: {} });
    seenRequests.length = 0;
    vi.spyOn(
      AgentServerConversationService,
      "batchGetAppConversations",
    ).mockResolvedValue([appConversation]);
    server.use(
      http.post(
        `${AGENT_SERVER}/api/conversations/:conversationId/ask_agent`,
        async ({ request, params }) => {
          seenRequests.push({
            method: request.method,
            pathname: new URL(request.url).pathname,
            body: await request.clone().json(),
          });
          expect(params.conversationId).toBe(CONV);
          return HttpResponse.json(SERVER_ERROR_BODY, { status: 500 });
        },
      ),
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("posts {question} to ask_agent and keeps the server error_id", async () => {
    const onSubmit = vi.fn();
    const { result } = renderHook(() => useBtwInterceptor(CONV, onSubmit));

    act(() => result.current(`/btw ${QUESTION}`));

    await waitFor(() => expect(entries()[0]?.status).toBe("error"));

    expect(onSubmit).not.toHaveBeenCalled();
    expect(seenRequests).toHaveLength(1);
    expect(seenRequests[0]).toMatchObject({
      method: "POST",
      pathname: `/api/conversations/${CONV}/ask_agent`,
      body: { question: QUESTION },
    });

    const response = entries()[0]?.response ?? "";
    expect(response).toContain(SERVER_ERROR_BODY.exception);
    expect(response).toContain(SERVER_ERROR_BODY.error_id);
  });
});
