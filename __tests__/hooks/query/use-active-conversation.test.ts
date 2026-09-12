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
import {
  UNTITLED_TERMINAL_FAST_POLL_ATTEMPTS,
  useActiveConversation,
} from "#/hooks/query/use-active-conversation";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";

// ── Mocks ─────────────────────────────────────────────────────────────────────

const {
  mockUseUserConversation,
  mockSetCurrentConversation,
  mockConversationId,
} = vi.hoisted(() => ({
  mockUseUserConversation: vi.fn(),
  mockSetCurrentConversation: vi.fn(),
  mockConversationId: { current: "conv-test" },
}));

vi.mock("#/hooks/query/use-user-conversation", () => ({
  useUserConversation: (...args: unknown[]) => mockUseUserConversation(...args),
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({
    conversationId: mockConversationId.current,
  }),
  useConversationId: () => ({ conversationId: mockConversationId.current }),
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
    mockConversationId.current = "conv-test";
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

  // ── give up fast-polling when autotitle never lands ────────────────────────
  //
  // isExecutionActive() includes IDLE and FINISHED, so an untitled conversation
  // that has already stopped executing would otherwise stay on the 3 s cadence
  // forever. Bound that branch; leave wake-up/resume (missing URL, PAUSED) and
  // still-running untitled conversations uncapped.

  function untitledQuery(
    executionStatus: ExecutionStatus,
    extra: Partial<AppConversation> = {},
  ) {
    return makeQuery({
      title: null,
      execution_status: executionStatus,
      conversation_url: "https://sandbox.example.com/api/conversations/conv-1",
      sandbox_status: "RUNNING",
      ...extra,
    });
  }

  it("falls back to 30000 after a bounded number of untitled FINISHED polls", () => {
    const intervalFn = renderAndCaptureIntervalFn();
    const query = untitledQuery(ExecutionStatus.FINISHED);

    const intervals = Array.from(
      { length: UNTITLED_TERMINAL_FAST_POLL_ATTEMPTS + 2 },
      () => intervalFn(query),
    );

    expect(intervals.slice(0, UNTITLED_TERMINAL_FAST_POLL_ATTEMPTS)).toEqual(
      Array(UNTITLED_TERMINAL_FAST_POLL_ATTEMPTS).fill(3000),
    );
    expect(intervals.slice(UNTITLED_TERMINAL_FAST_POLL_ATTEMPTS)).toEqual([
      30000, 30000,
    ]);
  });

  it("falls back to 30000 after a bounded number of untitled IDLE polls", () => {
    const intervalFn = renderAndCaptureIntervalFn();
    const query = untitledQuery(ExecutionStatus.IDLE);

    const intervals = Array.from(
      { length: UNTITLED_TERMINAL_FAST_POLL_ATTEMPTS + 1 },
      () => intervalFn(query),
    );

    expect(intervals.slice(0, UNTITLED_TERMINAL_FAST_POLL_ATTEMPTS)).toEqual(
      Array(UNTITLED_TERMINAL_FAST_POLL_ATTEMPTS).fill(3000),
    );
    expect(intervals[UNTITLED_TERMINAL_FAST_POLL_ATTEMPTS]).toBe(30000);
  });

  it("keeps fast-polling untitled RUNNING conversations with no attempt cap", () => {
    const intervalFn = renderAndCaptureIntervalFn();
    const query = untitledQuery(ExecutionStatus.RUNNING);

    for (let i = 0; i < UNTITLED_TERMINAL_FAST_POLL_ATTEMPTS + 5; i += 1) {
      expect(intervalFn(query)).toBe(3000);
    }
  });

  it("keeps fast-polling untitled WAITING_FOR_CONFIRMATION conversations with no attempt cap", () => {
    const intervalFn = renderAndCaptureIntervalFn();
    const query = untitledQuery(ExecutionStatus.WAITING_FOR_CONFIRMATION);

    for (let i = 0; i < UNTITLED_TERMINAL_FAST_POLL_ATTEMPTS + 5; i += 1) {
      expect(intervalFn(query)).toBe(3000);
    }
  });

  it("still fast-polls PAUSED sandboxes after the untitled-terminal give-up bound", () => {
    const intervalFn = renderAndCaptureIntervalFn();
    const query = untitledQuery(ExecutionStatus.FINISHED, {
      sandbox_status: "PAUSED",
    });

    for (let i = 0; i < UNTITLED_TERMINAL_FAST_POLL_ATTEMPTS + 5; i += 1) {
      expect(intervalFn(query)).toBe(3000);
    }
  });

  it("still fast-polls a missing conversation_url after the untitled-terminal give-up bound", () => {
    const intervalFn = renderAndCaptureIntervalFn();
    const query = untitledQuery(ExecutionStatus.FINISHED, {
      conversation_url: null,
    });

    for (let i = 0; i < UNTITLED_TERMINAL_FAST_POLL_ATTEMPTS + 5; i += 1) {
      expect(intervalFn(query)).toBe(3000);
    }
  });

  it("resumes fast-poll if execution becomes RUNNING after untitled-terminal give-up", () => {
    const intervalFn = renderAndCaptureIntervalFn();
    const finished = untitledQuery(ExecutionStatus.FINISHED);

    for (let i = 0; i < UNTITLED_TERMINAL_FAST_POLL_ATTEMPTS; i += 1) {
      expect(intervalFn(finished)).toBe(3000);
    }
    expect(intervalFn(finished)).toBe(30000);

    expect(intervalFn(untitledQuery(ExecutionStatus.RUNNING))).toBe(3000);
  });

  it("resets the untitled-terminal budget when the conversation id changes", () => {
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
    const { rerender } = renderHook(() => useActiveConversation());
    if (!captured) throw new Error("useUserConversation was not called");

    const first = untitledQuery(ExecutionStatus.FINISHED);
    for (let i = 0; i < UNTITLED_TERMINAL_FAST_POLL_ATTEMPTS; i += 1) {
      expect(captured(first)).toBe(3000);
    }
    expect(captured(first)).toBe(30000);

    mockConversationId.current = "conv-other";
    rerender();
    expect(captured(untitledQuery(ExecutionStatus.FINISHED))).toBe(3000);
  });

  it("walks RUNNING untitled → FINISHED untitled through fast-poll then give-up", () => {
    const intervalFn = renderAndCaptureIntervalFn();

    expect(intervalFn(untitledQuery(ExecutionStatus.RUNNING))).toBe(3000);
    expect(intervalFn(untitledQuery(ExecutionStatus.RUNNING))).toBe(3000);

    const finished = untitledQuery(ExecutionStatus.FINISHED);
    const finishedIntervals = Array.from(
      { length: UNTITLED_TERMINAL_FAST_POLL_ATTEMPTS + 1 },
      () => intervalFn(finished),
    );

    expect(
      finishedIntervals.slice(0, UNTITLED_TERMINAL_FAST_POLL_ATTEMPTS),
    ).toEqual(Array(UNTITLED_TERMINAL_FAST_POLL_ATTEMPTS).fill(3000));
    expect(finishedIntervals[UNTITLED_TERMINAL_FAST_POLL_ATTEMPTS]).toBe(30000);
  });
});
