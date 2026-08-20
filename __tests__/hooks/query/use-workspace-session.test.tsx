import React from "react";
import { RemoteWorkspace } from "@openhands/typescript-client/workspace/remote-workspace";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { callCloudProxy } from "#/api/cloud/proxy";
import {
  joinWorkspaceUrl,
  useWorkspaceSession,
} from "#/hooks/query/use-workspace-session";

// We mock the SDK workspace rather than the lower-level HttpClient:
// that's where our wiring contract lives (we hand the typescript-client a
// conversation id and trust it to do the right POST + return a base URL).
const startWorkspaceSessionMock = vi.fn();

vi.mock("@openhands/typescript-client/workspace/remote-workspace", () => ({
  RemoteWorkspace: vi.fn(function RemoteWorkspaceMock() {
    return {
      startWorkspaceSession: startWorkspaceSessionMock,
    };
  }),
}));

const callCloudProxyMock = vi.fn();
vi.mock("#/api/cloud/proxy", () => ({
  callCloudProxy: (...args: unknown[]) => callCloudProxyMock(...args),
}));

const getAgentServerClientOptionsMock = vi.fn();
vi.mock("#/api/agent-server-client-options", () => ({
  getAgentServerClientOptions: (...args: unknown[]) =>
    getAgentServerClientOptionsMock(...args),
}));

const getActiveBackendMock = vi.fn();
vi.mock("#/api/backend-registry/active-store", () => ({
  getActiveBackend: () => getActiveBackendMock(),
}));

const useActiveConversationMock = vi.fn();
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => useActiveConversationMock(),
}));

const useRuntimeIsReadyMock = vi.fn();
vi.mock("#/hooks/use-runtime-is-ready", () => ({
  useRuntimeIsReady: () => useRuntimeIsReadyMock(),
}));

function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const Wrapper = function WorkspaceSessionTestWrapper({
    children,
  }: {
    children: React.ReactNode;
  }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  };
  return Wrapper;
}

// Yields back to the event loop a few microtasks deep so react-query has a
// chance to schedule (and, in the negative-path tests, to NOT schedule) the
// query. ESLint forbids returning the timer id from `new Promise(...)`, so
// we wrap setTimeout in a void callback.
function flushScheduler(ms = 10): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

beforeEach(() => {
  startWorkspaceSessionMock.mockReset();
  callCloudProxyMock.mockReset();
  getAgentServerClientOptionsMock.mockReset();
  vi.mocked(RemoteWorkspace).mockClear();
  getActiveBackendMock.mockReset();
  useActiveConversationMock.mockReset();
  useRuntimeIsReadyMock.mockReset();
  useRuntimeIsReadyMock.mockReturnValue(true);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useWorkspaceSession", () => {
  describe("local backend", () => {
    it("calls startWorkspaceSession and exposes the returned baseUrl", async () => {
      getActiveBackendMock.mockReturnValue({
        backend: { id: "local-1", kind: "local", host: "http://localhost:8000" },
      });
      useActiveConversationMock.mockReturnValue({
        data: {
          id: "conv-1",
          conversation_url:
            "https://agent.example.com/api/conversations/conv-1",
          session_api_key: "key-abc",
        },
      });
      startWorkspaceSessionMock.mockResolvedValue(
        "https://agent.example.com/api/conversations/conv-1/workspace/",
      );
      getAgentServerClientOptionsMock.mockReturnValue({
        host: "http://agent.example.com",
        apiKey: "key-abc",
        workingDir: "workspace/project",
      });

      const { result } = renderHook(() => useWorkspaceSession(), {
        wrapper: makeWrapper(),
      });

      await waitFor(() => {
        expect(result.current.data?.baseUrl).toBe(
          "https://agent.example.com/api/conversations/conv-1/workspace/",
        );
      });

      expect(getAgentServerClientOptionsMock).toHaveBeenCalledTimes(1);
      expect(getAgentServerClientOptionsMock).toHaveBeenCalledWith({
        conversationUrl:
          "https://agent.example.com/api/conversations/conv-1",
        sessionApiKey: "key-abc",
      });
      expect(RemoteWorkspace).toHaveBeenCalledTimes(1);
      expect(RemoteWorkspace).toHaveBeenCalledWith({
        host: "http://agent.example.com",
        apiKey: "key-abc",
        workingDir: "workspace/project",
      });
      expect(startWorkspaceSessionMock).toHaveBeenCalledTimes(1);
      expect(startWorkspaceSessionMock).toHaveBeenCalledWith("conv-1");
      expect(callCloudProxyMock).not.toHaveBeenCalled();
    });
  });

  describe("cloud backend", () => {
    it("does not fire any request — workspace-session is local-only", async () => {
      getActiveBackendMock.mockReturnValue({
        backend: {
          id: "cloud-1",
          kind: "cloud",
          host: "https://app.all-hands.dev",
        },
      });
      useActiveConversationMock.mockReturnValue({
        data: {
          id: "conv-cloud",
          conversation_url:
            "https://abc123.prod-runtime.all-hands.dev/api/conversations/conv-cloud",
          session_api_key: "cloud-key-xyz",
        },
      });

      const { result } = renderHook(() => useWorkspaceSession(), {
        wrapper: makeWrapper(),
      });

      await flushScheduler();
      expect(callCloudProxyMock).not.toHaveBeenCalled();
      expect(startWorkspaceSessionMock).not.toHaveBeenCalled();
      expect(result.current.data).toBeNull();
    });
  });

  it("does not call startWorkspaceSession until the runtime is ready", async () => {
    getActiveBackendMock.mockReturnValue({
      backend: { id: "local-1", kind: "local", host: "http://localhost:8000" },
    });
    useActiveConversationMock.mockReturnValue({
      data: {
        id: "conv-1",
        conversation_url:
          "https://agent.example.com/api/conversations/conv-1",
        session_api_key: "key-abc",
      },
    });
    useRuntimeIsReadyMock.mockReturnValue(false);

    const { result } = renderHook(() => useWorkspaceSession(), {
      wrapper: makeWrapper(),
    });

    // Give react-query a tick to schedule (it shouldn't).
    await flushScheduler();
    expect(startWorkspaceSessionMock).not.toHaveBeenCalled();
    expect(callCloudProxyMock).not.toHaveBeenCalled();
    expect(result.current.data).toBeNull();
  });

  it("does not call startWorkspaceSession without a conversation id", async () => {
    getActiveBackendMock.mockReturnValue({
      backend: { id: "local-1", kind: "local", host: "http://localhost:8000" },
    });
    useActiveConversationMock.mockReturnValue({ data: undefined });

    renderHook(() => useWorkspaceSession(), { wrapper: makeWrapper() });

    await flushScheduler();
    expect(startWorkspaceSessionMock).not.toHaveBeenCalled();
    expect(callCloudProxyMock).not.toHaveBeenCalled();
  });
});

describe("joinWorkspaceUrl", () => {
  const base = "https://agent.example.com/api/conversations/c1/workspace/";

  it("returns the base URL when no relative path is supplied", () => {
    expect(joinWorkspaceUrl(base)).toBe(base);
    expect(joinWorkspaceUrl(base, "")).toBe(base);
    expect(joinWorkspaceUrl(base, null)).toBe(base);
  });

  it("appends a single-segment path", () => {
    expect(joinWorkspaceUrl(base, "index.html")).toBe(`${base}index.html`);
  });

  it("appends nested paths preserving separators", () => {
    expect(joinWorkspaceUrl(base, "src/components/App.tsx")).toBe(
      `${base}src/components/App.tsx`,
    );
  });

  it("strips leading slashes on the relative path", () => {
    expect(joinWorkspaceUrl(base, "/index.html")).toBe(`${base}index.html`);
    expect(joinWorkspaceUrl(base, "///deep/path.md")).toBe(
      `${base}deep/path.md`,
    );
  });

  it("URL-encodes individual segments but not the separators", () => {
    expect(joinWorkspaceUrl(base, "my files/has spaces.txt")).toBe(
      `${base}my%20files/has%20spaces.txt`,
    );
    expect(joinWorkspaceUrl(base, "tëst/résumé.pdf")).toBe(
      `${base}t%C3%ABst/r%C3%A9sum%C3%A9.pdf`,
    );
  });
});
