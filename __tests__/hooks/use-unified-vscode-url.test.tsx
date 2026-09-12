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
import { I18nKey } from "#/i18n/declaration";

const { mockUseConversationId } = vi.hoisted(() => ({
  mockUseConversationId: vi.fn(),
}));

vi.mock("react-i18next", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-i18next")>()),
  useTranslation: (namespace?: string) => ({
    t: (key: string) =>
      namespace === "openhands" ? key : `missing-namespace:${key}`,
  }),
}));

vi.mock("#/api/cloud/sandbox-service.api");
vi.mock("#/api/conversation-service/agent-server-conversation-service.api");
vi.mock("#/api/conversation-service/conversation-service.api");
vi.mock("#/contexts/active-backend-context");
vi.mock("#/hooks/query/use-active-conversation");
vi.mock("#/hooks/use-runtime-is-ready");
vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => mockUseConversationId(),
  useConversationId: () => mockUseConversationId(),
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

function makeSandbox(overrides: Partial<V1SandboxInfo> = {}): V1SandboxInfo {
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

function createQueryHarness() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, retryDelay: 0 } },
  });
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <I18nextProvider i18n={i18n}>{children}</I18nextProvider>
    </QueryClientProvider>
  );

  return { queryClient, wrapper };
}

function createWrapper() {
  return createQueryHarness().wrapper;
}

// The local path now gates the URL request on two capability checks the
// archived test predates:
//   1. `getOriginVSCodeBasePath()` — this browser origin must advertise an
//      editor route. In jsdom that comes from
//      `window.__AGENT_CANVAS_VSCODE_BASE_PATH__`, so we inject it here to
//      model a self-hosted deployment that serves the editor.
//   2. `AgentServerConversationService.getVSCodeStatus()` — a 200 probe that
//      must report `{ enabled: true, running: true }` before the URL query is
//      enabled. Default it to the available state so the existing local-mode
//      expectations still exercise the URL resolver.
const VSCODE_BASE_PATH = "/vscode";

beforeEach(() => {
  vi.resetAllMocks();
  (window as unknown as Record<string, unknown>).__AGENT_CANVAS_VSCODE_BASE_PATH__ =
    VSCODE_BASE_PATH;
  mockUseConversationId.mockReturnValue({ conversationId: "conv-123" });
  vi.mocked(useRuntimeIsReady).mockReturnValue(true);
  vi.mocked(useActiveConversation).mockReturnValue({
    data: makeConversation(),
  } as unknown as ReturnType<typeof useActiveConversation>);
  vi.mocked(AgentServerConversationService.getVSCodeStatus).mockResolvedValue({
    enabled: true,
    running: true,
  });
});

afterEach(() => {
  vi.clearAllMocks();
  delete (window as unknown as Record<string, unknown>)
    .__AGENT_CANVAS_VSCODE_BASE_PATH__;
});

describe("useUnifiedVSCodeUrl", () => {
  it("returns the cloud-computed VSCode URL from sandbox.exposed_urls in cloud mode", async () => {
    // Arrange — cloud backend, sandbox returned with a VSCODE entry.
    // This is the steady-state happy path: the cloud backend pre-builds the
    // public vscode subdomain URL and the GUI must surface it directly
    // instead of asking the runtime for /api/vscode/url (which only
    // knows its own localhost:8001).
    vi.mocked(useActiveBackend).mockReturnValue(cloudBackend);
    vi.mocked(batchGetCloudSandboxes).mockResolvedValue([
      makeSandbox({
        exposed_urls: [
          { name: "APP", url: "https://app.example.dev" },
          {
            name: "VSCODE",
            url: "https://vscode-abc.staging-runtime.all-hands.dev/?tkn=sek&folder=%2Fworkspace%2Fproject",
          },
        ],
      }),
    ]);

    // Act
    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    // Assert
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.url).toBe(
      "https://vscode-abc.staging-runtime.all-hands.dev/?tkn=sek&folder=%2Fworkspace%2Fproject",
    );
    expect(AgentServerConversationService.getVSCodeUrl).not.toHaveBeenCalled();
    // Cloud never runs the local capability probe, and the cloud path keeps
    // the control visible regardless of exposed_urls.
    expect(
      AgentServerConversationService.getVSCodeStatus,
    ).not.toHaveBeenCalled();
    expect(result.current.isUnavailable).toBe(false);
  });

  it("returns null url in cloud mode when the sandbox has no VSCODE exposed_url", async () => {
    // Arrange — sandbox is reachable but isn't running yet (STARTING /
    // PAUSED), so exposed_urls hasn't been populated. The hook must
    // surface "no URL" gracefully so the tab shows the empty-state
    // copy instead of crashing or serving a localhost fallback.
    vi.mocked(useActiveBackend).mockReturnValue(cloudBackend);
    vi.mocked(batchGetCloudSandboxes).mockResolvedValue([
      makeSandbox({
        status: "STARTING",
        exposed_urls: null,
      }),
    ]);

    // Act
    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    // Assert
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.url).toBeNull();
    expect(result.current.data?.error).toBe(
      i18n.t(I18nKey.VSCODE$URL_NOT_AVAILABLE),
    );
  });

  it("ignores unrelated cloud exposed URLs", async () => {
    vi.mocked(useActiveBackend).mockReturnValue(cloudBackend);
    vi.mocked(batchGetCloudSandboxes).mockResolvedValue([
      makeSandbox({
        exposed_urls: [{ name: "APP", url: "https://app.example.dev" }],
      }),
    ]);

    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual({
      url: null,
      error: i18n.t(I18nKey.VSCODE$URL_NOT_AVAILABLE),
    });
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
    expect(AgentServerConversationService.getVSCodeUrl).toHaveBeenCalledWith(
      "conv-123",
      "http://abc.staging-runtime.all-hands.dev/api/conv/1",
      "sek",
    );
    expect(batchGetCloudSandboxes).not.toHaveBeenCalled();
    expect(ConversationService.getVSCodeUrl).not.toHaveBeenCalled();
  });

  it("allows the VSCode query while the runtime reports an agent error", async () => {
    vi.mocked(useActiveBackend).mockReturnValue(localBackend);
    vi.mocked(useRuntimeIsReady).mockImplementation(
      (options) => options?.allowAgentError === true,
    );
    vi.mocked(AgentServerConversationService.getVSCodeUrl).mockResolvedValue({
      vscode_url: "https://vscode.example.dev/?folder=workspace",
    });

    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(useRuntimeIsReady).toHaveBeenCalledWith({ allowAgentError: true });
    expect(AgentServerConversationService.getVSCodeUrl).toHaveBeenCalledOnce();
  });

  it("stores local results under the conversation-specific cache key", async () => {
    vi.mocked(useActiveBackend).mockReturnValue(localBackend);
    vi.mocked(AgentServerConversationService.getVSCodeUrl).mockResolvedValue({
      vscode_url: "https://vscode.example.dev/?folder=workspace",
    });
    const { queryClient, wrapper } = createQueryHarness();

    const { result } = renderHook(() => useUnifiedVSCodeUrl(), { wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    const cachedQuery = queryClient.getQueryCache().find({
      exact: true,
      queryKey: [
        "unified",
        "vscode_url",
        "local",
        "conv-123",
        "http://abc.staging-runtime.all-hands.dev/api/conv/1",
        "sek",
      ],
    });
    expect(cachedQuery?.state.data).toEqual({
      url: "https://vscode.example.dev/?folder=workspace",
    });

    // The capability probe is cached under its own conversation-scoped key so a
    // different conversation can't reuse a stale enabled/running answer.
    const cachedStatusQuery = queryClient.getQueryCache().find({
      exact: true,
      queryKey: [
        "unified",
        "vscode_status",
        "local",
        "conv-123",
        "http://abc.staging-runtime.all-hands.dev/api/conv/1",
        "sek",
      ],
    });
    expect(cachedStatusQuery?.state.data).toEqual({
      enabled: true,
      running: true,
    });
  });

  it("refreshes stale local URL data when the hook remounts", async () => {
    vi.mocked(useActiveBackend).mockReturnValue(localBackend);
    vi.mocked(AgentServerConversationService.getVSCodeUrl)
      .mockResolvedValueOnce({
        vscode_url: "https://initial.example.dev/?folder=workspace",
      })
      .mockResolvedValueOnce({
        vscode_url: "https://remounted.example.dev/?folder=workspace",
      });
    const { wrapper } = createQueryHarness();

    const initial = renderHook(() => useUnifiedVSCodeUrl(), { wrapper });
    await waitFor(() => expect(initial.result.current.isSuccess).toBe(true));
    initial.unmount();

    const remounted = renderHook(() => useUnifiedVSCodeUrl(), { wrapper });
    await waitFor(() =>
      expect(remounted.result.current.data?.url).toBe(
        "https://remounted.example.dev/?folder=workspace",
      ),
    );
    expect(AgentServerConversationService.getVSCodeUrl).toHaveBeenCalledTimes(
      2,
    );
    // The probe is `refetchOnMount` too, so the remount re-verifies capability
    // rather than trusting the previous mount's cached answer.
    expect(
      AgentServerConversationService.getVSCodeStatus,
    ).toHaveBeenCalledTimes(2);
  });

  it("falls back to the legacy conversation service when the agent-server request fails", async () => {
    const agentServerFailure = new Error("runtime endpoint unavailable");
    vi.mocked(useActiveBackend).mockReturnValue(localBackend);
    vi.mocked(AgentServerConversationService.getVSCodeUrl).mockRejectedValue(
      agentServerFailure,
    );
    vi.mocked(ConversationService.getVSCodeUrl).mockResolvedValue({
      vscode_url: "https://fallback.example.dev/?folder=workspace",
    });

    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(ConversationService.getVSCodeUrl).toHaveBeenCalledWith("conv-123");
    expect(result.current.data).toEqual({
      url: "https://fallback.example.dev/?folder=workspace",
      error: null,
    });
  });

  it("returns local query failures after both URL services reject", async () => {
    const fallbackFailure = new Error("no VS Code endpoint");
    vi.mocked(useActiveBackend).mockReturnValue(localBackend);
    vi.mocked(AgentServerConversationService.getVSCodeUrl).mockRejectedValue(
      new Error("runtime unavailable"),
    );
    vi.mocked(ConversationService.getVSCodeUrl).mockRejectedValue(
      fallbackFailure,
    );

    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.status).toBe("error");
    expect(result.current.error).toBe(fallbackFailure);
    expect(result.current.data).toBeUndefined();
  });

  it("passes null runtime metadata when active conversation details are unavailable", async () => {
    vi.mocked(useActiveBackend).mockReturnValue(localBackend);
    vi.mocked(useActiveConversation).mockReturnValue({
      data: undefined,
    } as unknown as ReturnType<typeof useActiveConversation>);
    vi.mocked(AgentServerConversationService.getVSCodeUrl).mockResolvedValue({
      vscode_url: null,
    });

    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(AgentServerConversationService.getVSCodeUrl).toHaveBeenCalledWith(
      "conv-123",
      null,
      null,
    );
    expect(result.current.data).toEqual({
      url: null,
      error: i18n.t(I18nKey.VSCODE$URL_NOT_AVAILABLE),
    });
  });

  it("returns the refreshed local URL", async () => {
    vi.mocked(useActiveBackend).mockReturnValue(localBackend);
    vi.mocked(AgentServerConversationService.getVSCodeUrl)
      .mockResolvedValueOnce({
        vscode_url: "https://initial.example.dev/?folder=workspace",
      })
      .mockResolvedValueOnce({
        vscode_url: "https://refreshed.example.dev/?folder=workspace",
      });

    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    const refreshed = await result.current.refetch();

    expect(refreshed.data).toEqual({
      url: "https://refreshed.example.dev/?folder=workspace",
    });
    await waitFor(() =>
      expect(result.current.data?.url).toBe(
        "https://refreshed.example.dev/?folder=workspace",
      ),
    );
  });

  it("maps cloud refetches to refreshed, unavailable, and missing sandbox results", async () => {
    vi.mocked(useActiveBackend).mockReturnValue(cloudBackend);
    vi.mocked(batchGetCloudSandboxes)
      .mockResolvedValueOnce([makeSandbox()])
      .mockResolvedValueOnce([
        makeSandbox({
          exposed_urls: [
            { name: "APP", url: "https://app.example.dev" },
            { name: "VSCODE", url: "https://refreshed.example.dev" },
          ],
        }),
      ])
      .mockResolvedValueOnce([
        makeSandbox({
          exposed_urls: [{ name: "APP", url: "https://app.example.dev" }],
        }),
      ])
      .mockResolvedValueOnce([
        makeSandbox({
          exposed_urls: null,
        }),
      ])
      .mockResolvedValueOnce([]);

    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    const refreshed = await result.current.refetch();
    expect(refreshed.data).toEqual({ url: "https://refreshed.example.dev" });

    const unrelated = await result.current.refetch();
    expect(unrelated.data).toEqual({ url: null });

    const unavailable = await result.current.refetch();
    expect(unavailable.data).toEqual({ url: null });

    const missing = await result.current.refetch();
    expect(missing.data).toBeUndefined();
    await waitFor(() => expect(result.current.data?.url).toBeNull());
  });

  it("surfaces cloud sandbox lookup failures", async () => {
    const failure = new Error("sandbox lookup failed");
    vi.mocked(useActiveBackend).mockReturnValue(cloudBackend);
    vi.mocked(batchGetCloudSandboxes).mockRejectedValue(failure);

    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.status).toBe("error");
    expect(result.current.error).toBe(failure);
    expect(result.current.data).toBeUndefined();
  });

  it("does not request a cloud sandbox until the conversation has a sandbox id", () => {
    vi.mocked(useActiveBackend).mockReturnValue(cloudBackend);
    vi.mocked(useActiveConversation).mockReturnValue({
      data: makeConversation({ sandbox_id: null }),
    } as unknown as ReturnType<typeof useActiveConversation>);

    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    expect(batchGetCloudSandboxes).not.toHaveBeenCalled();
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
  });

  it("does not request a local URL before the runtime is ready", () => {
    vi.mocked(useActiveBackend).mockReturnValue(localBackend);
    vi.mocked(useRuntimeIsReady).mockReturnValue(false);

    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    expect(AgentServerConversationService.getVSCodeUrl).not.toHaveBeenCalled();
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
  });

  it("reports the missing conversation id when a disabled query is manually refreshed", async () => {
    vi.mocked(useActiveBackend).mockReturnValue(localBackend);
    mockUseConversationId.mockReturnValue({ conversationId: "" });

    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    expect(AgentServerConversationService.getVSCodeUrl).not.toHaveBeenCalled();
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();

    const refreshed = await result.current.refetch();

    expect(refreshed.data).toBeUndefined();
    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.error).toEqual(new Error("No conversation ID"));
    expect(AgentServerConversationService.getVSCodeUrl).not.toHaveBeenCalled();
  });

  it("hides the control and skips both requests when this origin serves no editor", async () => {
    // A public-mode / extra-backend origin advertises no editor route. The
    // agent-server might still report one, but this page couldn't reach it, so
    // neither the capability probe nor the URL request should fire and the
    // control must report itself unavailable rather than loading.
    vi.mocked(useActiveBackend).mockReturnValue(localBackend);
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_VSCODE_BASE_PATH__;

    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    expect(
      AgentServerConversationService.getVSCodeStatus,
    ).not.toHaveBeenCalled();
    expect(AgentServerConversationService.getVSCodeUrl).not.toHaveBeenCalled();
    expect(result.current.isUnavailable).toBe(true);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
  });

  it("hides the control when the capability probe reports the editor disabled", async () => {
    // `enabled: false` is the deployment switch. The URL request must never
    // fire — that is the whole point of gating on the probe — and the control
    // reports itself unavailable instead of erroring.
    vi.mocked(useActiveBackend).mockReturnValue(localBackend);
    vi.mocked(AgentServerConversationService.getVSCodeStatus).mockResolvedValue({
      enabled: false,
      running: true,
    });

    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isUnavailable).toBe(true));
    expect(AgentServerConversationService.getVSCodeUrl).not.toHaveBeenCalled();
    expect(result.current.isSuccess).toBe(false);
  });

  it("hides the control when the editor is enabled but not running", async () => {
    // `running: false` alongside `enabled: true` is a terminal failed-start,
    // not a startup race, so the URL request is still withheld.
    vi.mocked(useActiveBackend).mockReturnValue(localBackend);
    vi.mocked(AgentServerConversationService.getVSCodeStatus).mockResolvedValue({
      enabled: true,
      running: false,
    });

    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isUnavailable).toBe(true));
    expect(AgentServerConversationService.getVSCodeUrl).not.toHaveBeenCalled();
  });

  it("keeps the control available when the resolved URL is routed by this origin", async () => {
    // The probe cleared the editor and the resolved URL sits under the prefix
    // this origin actually serves, so the control is usable.
    const routedUrl = `${window.location.origin}/vscode/?tkn=abc&folder=workspace`;
    vi.mocked(useActiveBackend).mockReturnValue(localBackend);
    vi.mocked(AgentServerConversationService.getVSCodeUrl).mockResolvedValue({
      vscode_url: routedUrl,
    });

    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual({ url: routedUrl, error: null });
    expect(result.current.isUnavailable).toBe(false);
  });

  it("hides the control when the resolved URL is not routed by this origin", async () => {
    // Same-origin but wrong prefix: an extra backend with no prefix of its own
    // hands back a URL that would reopen this app instead of an editor. The URL
    // resolves successfully yet the control must still hide.
    const unroutableUrl = `${window.location.origin}/not-vscode/?tkn=abc`;
    vi.mocked(useActiveBackend).mockReturnValue(localBackend);
    vi.mocked(AgentServerConversationService.getVSCodeUrl).mockResolvedValue({
      vscode_url: unroutableUrl,
    });

    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.url).toBe(unroutableUrl);
    expect(result.current.isUnavailable).toBe(true);
  });

  it("marks the control unavailable when the editor reports no URL", async () => {
    // The probe says the editor is up, but the resolver returns no URL: there
    // is nothing to open, so the control hides rather than offering a no-op.
    vi.mocked(useActiveBackend).mockReturnValue(localBackend);
    vi.mocked(AgentServerConversationService.getVSCodeUrl).mockResolvedValue({
      vscode_url: null,
    });

    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual({
      url: null,
      error: i18n.t(I18nKey.VSCODE$URL_NOT_AVAILABLE),
    });
    expect(result.current.isUnavailable).toBe(true);
  });

  it("reports loading while the capability probe is in flight", () => {
    // The URL request only starts once the probe clears it, so the control is
    // "loading" for the probe as well — otherwise it would look ready while
    // there is still nothing to open.
    vi.mocked(useActiveBackend).mockReturnValue(localBackend);
    vi.mocked(AgentServerConversationService.getVSCodeUrl).mockResolvedValue({
      vscode_url: "https://vscode.example.dev/?folder=workspace",
    });

    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    expect(result.current.isLoading).toBe(true);
  });
});
