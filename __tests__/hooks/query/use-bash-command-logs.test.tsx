/**
 * Tests the error-tolerance logic in useBashCommandLogs.
 *
 * Cloud sandboxes can be paused, starting, errored, or deleted entirely
 * — and even when `sandbox_status` says "RUNNING" the runtime can be
 * unreachable for a moment. The hook is responsible for translating
 * each of these states into a stable `sandboxIssue` code so the modal
 * can render a targeted empty state instead of a raw axios error.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AxiosError, AxiosHeaders } from "axios";
import { HttpError } from "@openhands/typescript-client";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useBashCommandLogs } from "#/hooks/query/use-bash-command-logs";

// ── Mocks ─────────────────────────────────────────────────────────────────────

const { useUserConversationMock, useActiveBackendMock, listOutputsMock } =
  vi.hoisted(() => ({
    useUserConversationMock: vi.fn(),
    useActiveBackendMock: vi.fn(),
    listOutputsMock: vi.fn(),
  }));

vi.mock("#/hooks/query/use-user-conversation", () => ({
  useUserConversation: (...args: unknown[]) => useUserConversationMock(...args),
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => useActiveBackendMock(),
}));

vi.mock("#/api/bash-service/bash-service.api", () => ({
  default: { listOutputs: (...args: unknown[]) => listOutputsMock(...args) },
}));

// ── Helpers ───────────────────────────────────────────────────────────────────

function wrapper({ children }: { children: React.ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

function setActiveBackend(kind: "cloud" | "local") {
  useActiveBackendMock.mockReturnValue({
    backend: { id: "bk-1", kind },
    orgId: null,
  });
}

function setConversation(partial: Partial<AppConversation> | null) {
  useUserConversationMock.mockReturnValue({
    data: partial,
    isFetched: true,
    isPending: false,
  });
}

beforeEach(() => {
  useUserConversationMock.mockReset();
  useActiveBackendMock.mockReset();
  listOutputsMock.mockReset();
});

afterEach(() => {
  vi.clearAllMocks();
});

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("useBashCommandLogs — cloud sandbox state handling", () => {
  beforeEach(() => setActiveBackend("cloud"));

  it.each([
    { status: "MISSING" as const, issue: "missing" },
    { status: "PAUSED" as const, issue: "paused" },
    { status: "STARTING" as const, issue: "starting" },
    { status: "ERROR" as const, issue: "errored" },
  ])(
    "reports sandboxIssue=$issue and skips the fetch when sandbox_status=$status",
    async ({ status, issue }) => {
      setConversation({
        conversation_url: "https://runtime.example.com",
        sandbox_status: status,
      });

      const { result } = renderHook(
        () =>
          useBashCommandLogs({
            conversationId: "conv-1",
            bashCommandId: "cmd-1",
          }),
        { wrapper },
      );

      expect(result.current.sandboxIssue).toBe(issue);
      // Critical: we did NOT fire a doomed request at a known-bad sandbox.
      expect(listOutputsMock).not.toHaveBeenCalled();
    },
  );

  it("reports sandboxIssue=missing when conversation has no runtime URL", () => {
    setConversation({
      conversation_url: null,
      sandbox_status: "RUNNING",
    });

    const { result } = renderHook(
      () =>
        useBashCommandLogs({
          conversationId: "conv-1",
          bashCommandId: "cmd-1",
        }),
      { wrapper },
    );

    expect(result.current.sandboxIssue).toBe("missing");
    expect(listOutputsMock).not.toHaveBeenCalled();
  });

  it("reports conversationMissing when the lookup returns null", () => {
    setConversation(null);

    const { result } = renderHook(
      () =>
        useBashCommandLogs({
          conversationId: "conv-1",
          bashCommandId: "cmd-1",
        }),
      { wrapper },
    );

    expect(result.current.conversationMissing).toBe(true);
    expect(result.current.sandboxIssue).toBeNull();
    expect(listOutputsMock).not.toHaveBeenCalled();
  });

  it("fires the fetch and returns data when sandbox is RUNNING", async () => {
    setConversation({
      conversation_url: "https://runtime.example.com",
      sandbox_status: "RUNNING",
    });
    listOutputsMock.mockResolvedValueOnce([
      { kind: "BashOutput", stdout: "hi", stderr: null },
    ]);

    const { result } = renderHook(
      () =>
        useBashCommandLogs({
          conversationId: "conv-1",
          bashCommandId: "cmd-1",
        }),
      { wrapper },
    );

    await waitFor(() => expect(result.current.data).toBeTruthy());
    expect(result.current.sandboxIssue).toBeNull();
    expect(listOutputsMock).toHaveBeenCalledTimes(1);
  });

  it.each([
    { status: 404, label: "404 not-found" },
    { status: 502, label: "502 bad-gateway" },
    { status: 503, label: "503 unavailable" },
    { status: 504, label: "504 gateway-timeout" },
  ])(
    "classifies $label responses as sandboxIssue=unreachable",
    async ({ status }) => {
      setConversation({
        conversation_url: "https://runtime.example.com",
        sandbox_status: "RUNNING",
      });
      const headers = new AxiosHeaders();
      listOutputsMock.mockRejectedValueOnce(
        new AxiosError(
          `Request failed with status code ${status}`,
          "ERR",
          {
            headers,
          } as never,
          null,
          {
            status,
            data: {},
            statusText: "",
            headers,
            config: { headers },
          } as never,
        ),
      );

      const { result } = renderHook(
        () =>
          useBashCommandLogs({
            conversationId: "conv-1",
            bashCommandId: "cmd-1",
          }),
        { wrapper },
      );

      await waitFor(() =>
        expect(result.current.sandboxIssue).toBe("unreachable"),
      );
      // Don't double-report: when an issue is detected we hide the raw
      // error so the modal doesn't render both the empty state AND the
      // generic "Failed: AxiosError" line.
      expect(result.current.error).toBeNull();
    },
  );

  it("classifies network errors (no response) as sandboxIssue=unreachable", async () => {
    setConversation({
      conversation_url: "https://runtime.example.com",
      sandbox_status: "RUNNING",
    });
    const err = new AxiosError("Network Error", "ERR_NETWORK");
    err.response = undefined;
    listOutputsMock.mockRejectedValueOnce(err);

    const { result } = renderHook(
      () =>
        useBashCommandLogs({
          conversationId: "conv-1",
          bashCommandId: "cmd-1",
        }),
      { wrapper },
    );

    await waitFor(() =>
      expect(result.current.sandboxIssue).toBe("unreachable"),
    );
  });

  it("does NOT collapse 401/403 into unreachable (auth bugs surface as errors)", async () => {
    setConversation({
      conversation_url: "https://runtime.example.com",
      sandbox_status: "RUNNING",
    });
    const headers = new AxiosHeaders();
    listOutputsMock.mockRejectedValueOnce(
      new AxiosError("Forbidden", "ERR", { headers } as never, null, {
        status: 403,
        data: {},
        statusText: "Forbidden",
        headers,
        config: { headers },
      } as never),
    );

    const { result } = renderHook(
      () =>
        useBashCommandLogs({
          conversationId: "conv-1",
          bashCommandId: "cmd-1",
        }),
      { wrapper },
    );

    await waitFor(() => expect(result.current.error).toBeTruthy());
    expect(result.current.sandboxIssue).toBeNull();
  });

  it.each([
    { status: 404, label: "404 not-found" },
    { status: 503, label: "503 unavailable" },
  ])(
    "classifies shared-client HttpError $label responses as sandboxIssue=unreachable",
    async ({ status }) => {
      setConversation({
        conversation_url: "https://runtime.example.com",
        sandbox_status: "RUNNING",
      });
      listOutputsMock.mockRejectedValueOnce(
        new HttpError(status, "Error", { detail: "sandbox gone" }),
      );

      const { result } = renderHook(
        () =>
          useBashCommandLogs({
            conversationId: "conv-1",
            bashCommandId: "cmd-1",
          }),
        { wrapper },
      );

      await waitFor(() =>
        expect(result.current.sandboxIssue).toBe("unreachable"),
      );
      expect(result.current.error).toBeNull();
    },
  );

  it("does NOT collapse an HttpError 401 into unreachable (auth bugs surface as errors)", async () => {
    setConversation({
      conversation_url: "https://runtime.example.com",
      sandbox_status: "RUNNING",
    });
    listOutputsMock.mockRejectedValueOnce(
      new HttpError(401, "Unauthorized", { detail: "bad token" }),
    );

    const { result } = renderHook(
      () =>
        useBashCommandLogs({
          conversationId: "conv-1",
          bashCommandId: "cmd-1",
        }),
      { wrapper },
    );

    await waitFor(() => expect(result.current.error).toBeTruthy());
    expect(result.current.sandboxIssue).toBeNull();
  });

  it.each([
    {
      label: "network TypeError",
      makeError: () => new TypeError("Failed to fetch"),
    },
    {
      label: "TimeoutError abort",
      makeError: () =>
        Object.assign(new Error("The operation timed out"), {
          name: "TimeoutError",
        }),
    },
    {
      label: "timeout wrapped with cause",
      makeError: () =>
        new Error("Request timeout after 30000ms", {
          cause: Object.assign(new Error("timed out"), {
            name: "TimeoutError",
          }),
        }),
    },
  ])(
    "classifies fetch transport failures ($label) as sandboxIssue=unreachable",
    async ({ makeError }) => {
      setConversation({
        conversation_url: "https://runtime.example.com",
        sandbox_status: "RUNNING",
      });
      listOutputsMock.mockRejectedValueOnce(makeError());

      const { result } = renderHook(
        () =>
          useBashCommandLogs({
            conversationId: "conv-1",
            bashCommandId: "cmd-1",
          }),
        { wrapper },
      );

      await waitFor(() =>
        expect(result.current.sandboxIssue).toBe("unreachable"),
      );
    },
  );
});

describe("useBashCommandLogs — local backend", () => {
  beforeEach(() => setActiveBackend("local"));

  it("never reports a sandboxIssue (local has no sandbox lifecycle)", async () => {
    setConversation({
      conversation_url: null,
      sandbox_status: null,
    });
    listOutputsMock.mockResolvedValueOnce([]);

    const { result } = renderHook(
      () =>
        useBashCommandLogs({
          conversationId: "conv-1",
          bashCommandId: "cmd-1",
        }),
      { wrapper },
    );

    await waitFor(() => expect(result.current.data).toEqual([]));
    expect(result.current.sandboxIssue).toBeNull();
    // Even network/5xx errors stay as raw errors in local mode — they
    // typically mean the local agent-server is misconfigured, not a
    // sandbox lifecycle issue.
    expect(listOutputsMock).toHaveBeenCalledTimes(1);
  });
});
