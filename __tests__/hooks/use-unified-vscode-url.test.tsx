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

function createWrapper() {
  const queryClient = new QueryClient({
    // `retry` is overridden per-query by the hook's own `retry: 3`, so error
    // paths do retry here; `retryDelay: 0` keeps them from spending the
    // default exponential backoff before the query settles.
    defaultOptions: { queries: { retry: false, retryDelay: 0 } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <I18nextProvider i18n={i18n}>{children}</I18nextProvider>
    </QueryClientProvider>
  );
}

/** The prefix this origin serves the editor under, as static-server injects it. */
function advertiseEditorOnOrigin(basePath: string | null) {
  if (basePath === null) {
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_VSCODE_BASE_PATH__;
    return;
  }
  (
    window as unknown as Record<string, unknown>
  ).__AGENT_CANVAS_VSCODE_BASE_PATH__ = basePath;
}

/** An editor URL of the shape agent-server builds: this origin + its prefix. */
function editorUrlOnThisOrigin(basePath = "/vscode") {
  return `${window.location.origin}${basePath}/?tkn=local-key&folder=workspace`;
}

beforeEach(() => {
  vi.clearAllMocks();
  // Default to an origin that routes the editor. Availability is server
  // capability ∩ this origin's route table, and every pre-existing case here
  // is about the server half; the origin half has its own cases below.
  advertiseEditorOnOrigin("/vscode");
  vi.mocked(useRuntimeIsReady).mockReturnValue(true);
  vi.mocked(useActiveConversation).mockReturnValue({
    data: makeConversation(),
  } as unknown as ReturnType<typeof useActiveConversation>);
  // Default the capability probe to "editor present and running". Local-mode
  // tests that care about the capability state override this; the rest would
  // otherwise never reach the URL request, which is gated on it.
  vi.mocked(AgentServerConversationService.getVSCodeStatus).mockResolvedValue({
    enabled: true,
    running: true,
  });
});

afterEach(() => {
  vi.clearAllMocks();
  advertiseEditorOnOrigin(null);
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
    expect(AgentServerConversationService.getVSCodeUrl).not.toHaveBeenCalled();
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
    // Cloud is deliberately excluded from `isUnavailable`: a sandbox that is
    // still STARTING will populate exposed_urls shortly, so the control stays
    // visible. Only self-hosted backends can report a final "no editor".
    expect(result.current.isUnavailable).toBe(false);
  });

  it("falls through to AgentServerConversationService.getVSCodeUrl in local mode", async () => {
    // Arrange — local backend: cloud sandbox lookup must be skipped and
    // the existing local resolver must drive the URL. Regression check
    // for the cloud/local branch that was added to the hook.
    vi.mocked(useActiveBackend).mockReturnValue(localBackend);
    vi.mocked(AgentServerConversationService.getVSCodeUrl).mockResolvedValue({
      vscode_url: editorUrlOnThisOrigin(),
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
    // A backend that hands back a usable URL is available, so consumers
    // render the control.
    expect(result.current.isUnavailable).toBe(false);
  });

  it("reports isUnavailable in local mode when the backend has VSCode disabled", async () => {
    // Arrange — `enable_vscode: false`. The capability probe answers 200 with
    // `enabled: false`, so this is a value rather than an error: the control
    // is dropped without the URL request ever running, which is what keeps
    // the 503 from `/vscode/url` (and its toast) off the screen entirely.
    vi.mocked(useActiveBackend).mockReturnValue(localBackend);
    vi.mocked(AgentServerConversationService.getVSCodeStatus).mockResolvedValue(
      {
        enabled: false,
        running: false,
        message: "VSCode is disabled in configuration",
      },
    );

    // Act
    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    // Assert
    await waitFor(() => expect(result.current.isUnavailable).toBe(true));
    expect(result.current.isError).toBe(false);
    expect(AgentServerConversationService.getVSCodeUrl).not.toHaveBeenCalled();
    expect(ConversationService.getVSCodeUrl).not.toHaveBeenCalled();
  });

  it("reports isUnavailable in local mode when the editor is enabled but not running", async () => {
    // Arrange — configured, but the process failed to start (or has died).
    // agent-server awaits VSCodeService.start() in its lifespan before it
    // serves any request, so `running: false` here is terminal rather than a
    // startup window. `/vscode/url` would still hand back a URL, so this
    // state is only visible through the probe.
    vi.mocked(useActiveBackend).mockReturnValue(localBackend);
    vi.mocked(AgentServerConversationService.getVSCodeStatus).mockResolvedValue(
      {
        enabled: true,
        running: false,
      },
    );

    // Act
    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    // Assert
    await waitFor(() => expect(result.current.isUnavailable).toBe(true));
    expect(result.current.isError).toBe(false);
    expect(AgentServerConversationService.getVSCodeUrl).not.toHaveBeenCalled();
  });

  it("keeps a failing capability probe observable as an error rather than hiding the control", async () => {
    // Arrange — a transport, auth or server fault on the probe itself. This
    // is the case the previous `isError`-derived `isUnavailable` conflated
    // with a disabled editor: it must stay an error (so retry and the global
    // toast still apply) and must NOT silently remove the control, because
    // nothing here says the deployment has no editor.
    vi.mocked(useActiveBackend).mockReturnValue(localBackend);
    vi.mocked(AgentServerConversationService.getVSCodeStatus).mockRejectedValue(
      new Error("Request failed with status code 401"),
    );

    // Act
    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    // Assert
    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.isUnavailable).toBe(false);
  });

  it("reports isUnavailable in local mode when the backend reports no URL", async () => {
    // Arrange — the probe reports a running editor, but the URL request
    // carries no URL to point at (e.g. no connection token). Distinct from
    // both cases above: the capability state is fine and the query settles in
    // `success`, so neither the probe nor `isError` would catch it.
    vi.mocked(useActiveBackend).mockReturnValue(localBackend);
    vi.mocked(AgentServerConversationService.getVSCodeUrl).mockResolvedValue({
      vscode_url: null,
    });

    // Act
    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    // Assert
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.url).toBeNull();
    expect(result.current.isUnavailable).toBe(true);
  });

  it("reports isUnavailable when this origin serves no editor (public mode)", async () => {
    // Arrange — docker's public-mode static server shares one agent-server
    // with the main instance but deliberately omits the editor route, because
    // the editor's connection token is the session API key and that origin
    // exists to test the unauthenticated case. The shared agent-server still
    // answers `enabled: true, running: true`, so a control gated on the probe
    // alone renders here and then falls through to the SPA.
    vi.mocked(useActiveBackend).mockReturnValue(localBackend);
    advertiseEditorOnOrigin(null);

    // Act
    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    // Assert — hidden, and cheaply: neither request is worth making.
    await waitFor(() => expect(result.current.isUnavailable).toBe(true));
    expect(result.current.isError).toBe(false);
    expect(
      AgentServerConversationService.getVSCodeStatus,
    ).not.toHaveBeenCalled();
    expect(AgentServerConversationService.getVSCodeUrl).not.toHaveBeenCalled();
  });

  it("reports isUnavailable when the URL resolves outside this origin's editor route", async () => {
    // Arrange — a conversation on an extra backend (dev-extra-backend.mjs),
    // registered from a browser whose origin belongs to the bundled stack.
    // That backend configures no prefix of its own, so agent-server appends
    // nothing to the origin we send it and hands back the canvas root. Same
    // origin, and the probe is truthful about *that* server — but clicking
    // would reopen this app, or reach the bundled stack's editor and hence a
    // different container's workspace.
    vi.mocked(useActiveBackend).mockReturnValue(localBackend);
    vi.mocked(AgentServerConversationService.getVSCodeUrl).mockResolvedValue({
      vscode_url: `${window.location.origin}/?tkn=extra-backend-key&folder=workspace`,
    });

    // Act
    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    // Assert
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.isUnavailable).toBe(true);
  });

  it("renders the control when the URL lands under this origin's editor route", async () => {
    // Arrange — the bundled stack: the origin advertises `/vscode` because it
    // routes `/vscode`, and the conversation's own agent-server is configured
    // with the matching prefix. This is the case the whole feature exists for,
    // pinned here so the guards above cannot be tightened into hiding it.
    vi.mocked(useActiveBackend).mockReturnValue(localBackend);
    vi.mocked(AgentServerConversationService.getVSCodeUrl).mockResolvedValue({
      vscode_url: editorUrlOnThisOrigin("/vscode"),
    });

    // Act
    const { result } = renderHook(() => useUnifiedVSCodeUrl(), {
      wrapper: createWrapper(),
    });

    // Assert
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.isUnavailable).toBe(false);
    expect(result.current.data?.url).toBe(editorUrlOnThisOrigin("/vscode"));
  });
});
