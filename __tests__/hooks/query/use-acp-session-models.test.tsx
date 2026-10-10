import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useAcpSessionModels } from "#/hooks/query/use-acp-session-models";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import {
  acpModelScope,
  readRememberedAcpModels,
} from "#/utils/remembered-acp-models";

const backendMock = vi.hoisted(() => ({
  current: { backend: { id: "cloud-1", kind: "cloud" as "local" | "cloud" } },
}));
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => backendMock.current,
}));

const getRuntimeConversation = vi.hoisted(() => vi.fn());
vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({ default: { getRuntimeConversation } }),
);

const ALL = acpModelScope(null);

const MODELS = [
  { id: "default", label: "Default (recommended)" },
  { id: "sonnet", label: "Sonnet" },
];

function conversation(overrides: Partial<AppConversation> = {}) {
  return {
    id: "conv-1",
    acp_server: "claude-code",
    conversation_url: "https://runtime.example.com/api/conversations/conv-1",
    session_api_key: "session-key",
    sandbox_status: "RUNNING",
    acp_available_models: null,
    ...overrides,
  } as AppConversation;
}

function wrapper({ children }: { children: React.ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

beforeEach(() => {
  vi.clearAllMocks();
  window.localStorage.clear();
  backendMock.current = { backend: { id: "cloud-1", kind: "cloud" } };
});

describe("useAcpSessionModels", () => {
  it("reads a cloud conversation's list from its sandbox and remembers it", async () => {
    getRuntimeConversation.mockResolvedValue({ available_models: MODELS });

    const { result } = renderHook(() => useAcpSessionModels(conversation()), {
      wrapper,
    });

    await waitFor(() => expect(result.current).toEqual(MODELS));
    expect(getRuntimeConversation).toHaveBeenCalledWith(
      "conv-1",
      "https://runtime.example.com/api/conversations/conv-1",
      "session-key",
    );
    expect(readRememberedAcpModels("cloud-1", "claude-code", ALL)).toEqual(
      MODELS,
    );
  });

  it("waits for the sandbox to run", () => {
    const { result } = renderHook(
      () => useAcpSessionModels(conversation({ sandbox_status: "PAUSED" })),
      { wrapper },
    );

    expect(getRuntimeConversation).not.toHaveBeenCalled();
    expect(result.current).toEqual([]);
  });

  it("uses a local conversation's own list without asking the runtime", () => {
    backendMock.current = { backend: { id: "local-1", kind: "local" } };

    const { result } = renderHook(
      () => useAcpSessionModels(conversation({ acp_available_models: MODELS })),
      { wrapper },
    );

    expect(result.current).toEqual(MODELS);
    expect(getRuntimeConversation).not.toHaveBeenCalled();
    expect(readRememberedAcpModels("local-1", "claude-code", ALL)).toEqual(
      MODELS,
    );
  });

  it("remembers a list under the secrets the conversation launched with", () => {
    backendMock.current = { backend: { id: "local-1", kind: "local" } };
    const launched_agent_profile = {
      agent_profile_id: "id-claude",
      revision: 2,
      secret_refs: ["ANTHROPIC_API_KEY"],
    };

    renderHook(
      () =>
        useAcpSessionModels(
          conversation({
            acp_available_models: MODELS,
            launched_agent_profile,
          }),
        ),
      { wrapper },
    );

    expect(
      readRememberedAcpModels(
        "local-1",
        "claude-code",
        acpModelScope(["ANTHROPIC_API_KEY"]),
      ),
    ).toEqual(MODELS);
    expect(readRememberedAcpModels("local-1", "claude-code", ALL)).toEqual([]);
  });

  it("does not remember a custom server's list", () => {
    backendMock.current = { backend: { id: "local-1", kind: "local" } };

    renderHook(
      () =>
        useAcpSessionModels(
          conversation({ acp_server: "custom", acp_available_models: MODELS }),
        ),
      { wrapper },
    );

    expect(readRememberedAcpModels("local-1", "custom")).toEqual([]);
  });
});
