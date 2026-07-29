/**
 * Tests that useActiveConversation passes the correct refetchInterval callback
 * to useUserConversation.
 *
 * Regression: the original callback only fast-polled (3 s) when
 * conversation_url was absent. For PAUSED sandboxes the cloud API keeps the
 * old conversation_url — checking that field alone left the hook on the slow
 * 30-second interval while the sandbox was waking up after a resume call.
 *
 * The fix adds sandbox_status === "PAUSED" as a second fast-poll trigger so
 * the hook picks up the PAUSED → RUNNING transition within ~3 s regardless of
 * whether conversation_url is present.
 *
 * It also fast-polls while the title is still unset and the agent is actively
 * executing, so the header title (which lands asynchronously) refreshes within
 * ~3 s instead of on the slow 30 s tick.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook } from "@testing-library/react";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";

// ── Mocks ─────────────────────────────────────────────────────────────────────

const { mockUseUserConversation, mockSetCurrentConversation } = vi.hoisted(
  () => ({
    mockUseUserConversation: vi.fn(),
    mockSetCurrentConversation: vi.fn(),
  }),
);

vi.mock("#/hooks/query/use-user-conversation", () => ({
  useUserConversation: (...args: unknown[]) => mockUseUserConversation(...args),
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: "conv-test" }),
  useConversationId: () => ({ conversationId: "conv-test" }),
}));

vi.mock("#/api/conversation-service/conversation-service.api", () => ({
  default: { setCurrentConversation: mockSetCurrentConversation },
}));

// ── Helpers ───────────────────────────────────────────────────────────────────

type IntervalFn = (query: {
  state: { data: AppConversation | null | undefined };
}) => number;

/** Render the hook and return the refetchInterval function it passed to useUserConversation. */
function renderAndCaptureIntervalFn(): IntervalFn {
  let captured: IntervalFn | undefined;

  mockUseUserConversation.mockImplementation(
    (_cid: string | null, intervalFn: IntervalFn) => {
      captured = intervalFn;
      return {
        data: undefined,
        isLoading: false,
        isPending: false,
        isFetched: false,
        error: null,
        isError: false,
      };
    },
  );

  renderHook(() => useActiveConversation());

  if (!captured) throw new Error("useUserConversation was not called");
  return captured;
}

function makeQuery(data: Partial<AppConversation> | null | undefined): {
  state: { data: AppConversation | null | undefined };
} {
  if (!data) return { state: { data: data as null | undefined } };
  return {
    state: {
      data: {
        id: "conv-1",
        created_by_user_id: null,
        selected_repository: null,
        selected_branch: null,
        git_provider: null,
        title: "Test",
        trigger: null,
        pr_number: [],
        llm_model: null,
        metrics: null,
        created_at: "2024-01-01T00:00:00Z",
        updated_at: "2024-01-01T00:00:00Z",
        execution_status: null,
        conversation_url:
          "https://sandbox.example.com/api/conversations/conv-1",
        session_api_key: null,
        sandbox_id: null,
        sub_conversation_ids: [],
        ...data,
      } as AppConversation,
    },
  };
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("useActiveConversation — refetchInterval callback", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("returns 3000 when sandbox_status is PAUSED (even if conversation_url is present)", () => {
    const intervalFn = renderAndCaptureIntervalFn();

    const result = intervalFn(
      makeQuery({
        sandbox_status: "PAUSED",
        conversation_url:
          "https://sandbox.example.com/api/conversations/conv-1",
      }),
    );

    expect(result).toBe(3000);
  });

  it("returns 3000 when conversation_url is null (sandbox still starting)", () => {
    const intervalFn = renderAndCaptureIntervalFn();

    const result = intervalFn(makeQuery({ conversation_url: null }));

    expect(result).toBe(3000);
  });

  it("returns 30000 when conversation_url is present and sandbox_status is null (local backend)", () => {
    const intervalFn = renderAndCaptureIntervalFn();

    const result = intervalFn(
      makeQuery({ sandbox_status: null, conversation_url: "https://..." }),
    );

    expect(result).toBe(30000);
  });

  it("returns 30000 when conversation_url is present and sandbox_status is RUNNING", () => {
    const intervalFn = renderAndCaptureIntervalFn();

    const result = intervalFn(
      makeQuery({
        sandbox_status: "RUNNING",
        conversation_url:
          "https://sandbox.example.com/api/conversations/conv-1",
      }),
    );

    expect(result).toBe(30000);
  });

  it("returns 30000 when query data is null (conversation not yet loaded)", () => {
    const intervalFn = renderAndCaptureIntervalFn();

    // data is null / undefined — the `if (data && ...)` guard returns false
    expect(intervalFn(makeQuery(null))).toBe(30000);
    expect(intervalFn(makeQuery(undefined))).toBe(30000);
  });

  // ── fast-poll until the title is set ───────────────────────────────────────

  it("returns 3000 when title is unset and the agent is actively executing", () => {
    const intervalFn = renderAndCaptureIntervalFn();

    const result = intervalFn(
      makeQuery({
        title: null,
        execution_status: ExecutionStatus.RUNNING,
        conversation_url:
          "https://sandbox.example.com/api/conversations/conv-1",
        sandbox_status: "RUNNING",
      }),
    );

    expect(result).toBe(3000);
  });

  it("returns 3000 when title is an empty string and the agent is actively executing", () => {
    const intervalFn = renderAndCaptureIntervalFn();

    const result = intervalFn(
      makeQuery({
        title: "",
        execution_status: ExecutionStatus.IDLE,
        conversation_url:
          "https://sandbox.example.com/api/conversations/conv-1",
        sandbox_status: "RUNNING",
      }),
    );

    expect(result).toBe(3000);
  });

  it("returns 30000 when title is set, even while the agent is executing", () => {
    const intervalFn = renderAndCaptureIntervalFn();

    const result = intervalFn(
      makeQuery({
        title: "Generated title",
        execution_status: ExecutionStatus.RUNNING,
        conversation_url:
          "https://sandbox.example.com/api/conversations/conv-1",
        sandbox_status: "RUNNING",
      }),
    );

    expect(result).toBe(30000);
  });

  it("returns 30000 when title is unset but execution has errored (no title is coming)", () => {
    const intervalFn = renderAndCaptureIntervalFn();

    const result = intervalFn(
      makeQuery({
        title: null,
        execution_status: ExecutionStatus.ERROR,
        conversation_url:
          "https://sandbox.example.com/api/conversations/conv-1",
        sandbox_status: "RUNNING",
      }),
    );

    expect(result).toBe(30000);
  });

  it("returns 30000 when title is unset and execution_status is null (idle/terminal, not active)", () => {
    const intervalFn = renderAndCaptureIntervalFn();

    const result = intervalFn(
      makeQuery({
        title: null,
        execution_status: null,
        conversation_url:
          "https://sandbox.example.com/api/conversations/conv-1",
        sandbox_status: "RUNNING",
      }),
    );

    expect(result).toBe(30000);
  });
});
