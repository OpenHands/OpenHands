import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { I18nextProvider } from "react-i18next";
import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import React from "react";
import { useUnifiedVSCodeUrl } from "#/hooks/query/use-unified-vscode-url";
import { batchGetCloudSandboxes } from "#/api/cloud/sandbox-service.api";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";
import type { ResolvedActiveBackend } from "#/api/backend-registry/types";
import type { V1SandboxInfo } from "#/api/cloud/sandbox-service.types";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";

vi.mock("#/api/cloud/sandbox-service.api");
vi.mock("#/api/conversation-service/agent-server-conversation-service.api");
vi.mock("#/api/conversation-service/conversation-service.api");
vi.mock("#/contexts/active-backend-context");
vi.mock("#/hooks/query/use-active-conversation");
vi.mock("#/hooks/use-runtime-is-ready");
vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: "test-conversation-id" }),
  useConversationId: () => ({ conversationId: "conv-123" }),
}));

if (!i18n.isInitialized) {
  i18n.use(initReactI18next).init({
    lng: "en",
    fallbackLng: "en",
    ns: ["openhands"],
    defaultNS: "openhands",
    resources: { en: { openhands: {} } },
    interpolation: { escapeValue: false },
    returnEmptyString: false,
  });
}

const cloudBackend: ResolvedActiveBackend = {
  backend: {
    id: "cloud-prod",
    name: "Production",
    host: "https://app.all-hands.dev",
    apiKey: "key",
    kind: "cloud",
  },
  orgId: "org-1",
};

const localBackend: ResolvedActiveBackend = {
  backend: {
    id: "local-1",
    name: "Local",
    host: "http://localhost:8000",
    apiKey: "key",
    kind: "local",
  },
  orgId: null,
};

function makeConversation(
  overrides: Partial<AppConversation> = {},
): AppConversation {
  return {
    id: "conv-123",
    sandbox_id: "sandbox-9",
    conversation_url: "http://abc.staging-runtime.all-hands.dev/api/conv/1",
    session_api_key: "sek",
    created_by_user_id: null,
    selected_repository: null,
    selected_branch: null,
    git_provider: null,
    title: null,
    trigger: null,
    pr_number: [],
    llm_model: null,
    metrics: null,
    created_at: "2026-05-12T00:00:00Z",
    updated_at: "2026-05-12T00:00:00Z",
    execution_status: "running",
    sub_conversation_ids: [],
    ...overrides,
  } as AppConversation;
}

function makeSandbox(
  overrides: Partial<V1SandboxInfo> = {},
): V1SandboxInfo {
  return {
    id: "sandbox-9",
    created_by_user_id: null,
    sandbox_spec_id: "spec-1",
    status: "RUNNING",
    session_api_key: "sek",
    exposed_urls: [
      {
        name: "VSCODE",
        url: "https://vscode-abc.staging-runtime.all-hands.dev/?tkn=sek&folder=%2Fworkspace%2Fproject",
      },
    ],
    created_at: "2026-05-12T00:00:00Z",
    ...overrides,
  };
}

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <I18nextProvider i18n={i18n}>{children}</I18nextProvider>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(useRuntimeIsReady).mockReturnValue(true);
  vi.mocked(useActiveConversation).mockReturnValue({
    data: makeConversation(),
  } as unknown as ReturnType<typeof useActiveConversation>);
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("useUnifiedVSCodeUrl", () => {
  it("returns the cloud-computed VSCode URL from sandbox.exposed_urls in cloud mode", async () => {
    // Arrange — cloud backend, sandbox returned with a VSCODE entry.
    // This is the steady-state happy path: the cloud backend pre-builds the
    // public vscode subdomain URL and the GUI must surface it directly
    // instead of asking the runtime for /api/vscode/url (which only
    // knows its own localhost:8001).
    vi.mocked(useActiveBackend).mockReturnValue(cloudBackend);
    vi.mocked(batchGetCloudSandboxes).mockResolvedValue([makeSandbox()]);

    // Act
    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    // Assert
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.url).toBe(
      "https://vscode-abc.staging-runtime.all-hands.dev/?tkn=sek&folder=%2Fworkspace%2Fproject",
    );
    expect(
      AgentServerConversationService.getVSCodeUrl,
    ).not.toHaveBeenCalled();
  });

  it("returns null url in cloud mode when the sandbox has no VSCODE exposed_url", async () => {
    // Arrange — sandbox is reachable but isn't running yet (STARTING /
    // PAUSED), so exposed_urls hasn't been populated. The hook must
    // surface "no URL" gracefully so the tab shows the empty-state
    // copy instead of crashing or serving a localhost fallback.
    vi.mocked(useActiveBackend).mockReturnValue(cloudBackend);
    vi.mocked(batchGetCloudSandboxes).mockResolvedValue([
      makeSandbox({ status: "STARTING", exposed_urls: null }),
    ]);

    // Act
    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    // Assert
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.url).toBeNull();
  });

  it("falls through to AgentServerConversationService.getVSCodeUrl in local mode", async () => {
    // Arrange — local backend: cloud sandbox lookup must be skipped and
    // the existing local resolver must drive the URL. Regression check
    // for the cloud/local branch that was added to the hook.
    vi.mocked(useActiveBackend).mockReturnValue(localBackend);
    vi.mocked(AgentServerConversationService.getVSCodeUrl).mockResolvedValue({
      vscode_url: "http://localhost:8001/?tkn=local-key&folder=workspace",
    });

    // Act
    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    // Assert
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(
      AgentServerConversationService.getVSCodeUrl,
    ).toHaveBeenCalledWith(
      "conv-123",
      "http://abc.staging-runtime.all-hands.dev/api/conv/1",
      "sek",
    );
    expect(batchGetCloudSandboxes).not.toHaveBeenCalled();
    expect(ConversationService.getVSCodeUrl).not.toHaveBeenCalled();
  });
});
