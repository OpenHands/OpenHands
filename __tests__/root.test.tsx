import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createRoutesStub } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { http, HttpResponse } from "msw";
import App, { links } from "#/root";
import { server } from "#/mocks/node";
import {
  __resetActiveStoreForTests,
  NO_BACKEND_ID,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { ONBOARDING_COMPLETED_STORAGE_KEY } from "#/components/features/onboarding/use-onboarding-completion";
import { ONBOARDING_DISMISSED_SESSION_KEY_PREFIX } from "#/components/features/onboarding/use-onboarding-dismissal";
import { SEEDED_DEFAULT_BACKEND_ID } from "#/api/backend-registry/default-backend";

const TRANSLATIONS: Record<string, string> = {
  BACKEND$MANAGE_TITLE: "Manage backends",
  BACKEND$RECONNECT_CLOUD_TITLE: "Reconnect to Cloud",
  BACKEND$RECONNECT_CLOUD: "Reconnect to Cloud",
  BACKEND$MANAGE_EMPTY: "No backends yet.",
  BACKEND$ADD: "+ Add Backend",
  BACKEND$LOG_BACK_IN: "Log back in",
  BACKEND$LOGGED_OUT: "Logged out",
  BACKEND$KIND_LOCAL: "Local",
  BACKEND$KIND_CLOUD: "Cloud",
  BACKEND$EDIT: "Edit",
  BACKEND$REMOVE: "Remove",
  HOME$DONE: "Done",
};

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: Record<string, string | number>) => {
      let value = TRANSLATIONS[key] ?? key;
      for (const [optionKey, optionValue] of Object.entries(options ?? {})) {
        value = value.replaceAll(`{{${optionKey}}}`, String(optionValue));
      }
      return value;
    },
  }),
}));

vi.mock("#/components/features/onboarding/onboarding-modal", async () => {
  const React = await import("react");
  const { useNavigation } = await import("#/context/navigation-context");

  return {
    OnboardingModal: ({
      onClose,
      initialStep,
      isPreview,
    }: {
      onClose: () => void;
      initialStep?: number;
      isPreview?: boolean;
    }) => {
      const { navigate } = useNavigation();
      return React.createElement(
        "div",
        {
          "data-testid": "onboarding-modal",
          "data-initial-step": initialStep,
          "data-is-preview": String(isPreview ?? false),
        },
        React.createElement("div", {
          "data-testid": "onboarding-step-check-backend",
        }),
        React.createElement(
          "button",
          {
            type: "button",
            "data-testid": "mock-onboarding-launch",
            onClick: () => {
              navigate("/conversations/mock-conversation");
              onClose();
            },
          },
          "Launch conversation",
        ),
      );
    },
  };
});

const ORIGINAL_LOCATION = window.location;

const RouterStub = createRoutesStub([
  {
    Component: App,
    path: "/",
    children: [
      {
        Component: () => <div data-testid="app-outlet">app outlet</div>,
        path: "/",
      },
      {
        Component: () => (
          <div data-testid="conversation-outlet">conversation outlet</div>
        ),
        path: "/conversations/:conversationId",
      },
    ],
  },
]);

const renderApp = (initialEntries: string[] = ["/"]) =>
  render(<RouterStub initialEntries={initialEntries} />, {
    wrapper: ({ children }) => (
      <QueryClientProvider
        client={
          new QueryClient({
            defaultOptions: { queries: { retry: false } },
          })
        }
      >
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    ),
  });

function dismissOnboardingForBackend(backendId = SEEDED_DEFAULT_BACKEND_ID) {
  window.sessionStorage.setItem(
    `${ONBOARDING_DISMISSED_SESSION_KEY_PREFIX}:${backendId}`,
    "1",
  );
}
describe("App root agent-server availability guard", () => {
  beforeEach(() => {
    window.localStorage.clear();
    window.sessionStorage.clear();
    vi.unstubAllEnvs();
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_AUTH_REQUIRED__;
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_LOCK_TO_CLOUD__;
    (
      window as unknown as Record<string, unknown>
    ).__AGENT_CANVAS_SESSION_API_KEY__ = "test-session-key";
    __resetActiveStoreForTests();
  });

  afterEach(() => {
    Object.defineProperty(window, "location", {
      configurable: true,
      value: ORIGINAL_LOCATION,
    });
  });

  it("shows first-run onboarding before the auth gate when public mode has no backend key", async () => {
    vi.stubEnv("VITE_AUTH_REQUIRED", "true");
    vi.stubEnv("VITE_SESSION_API_KEY", "");
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_SESSION_API_KEY__;
    window.localStorage.clear();
    __resetActiveStoreForTests();

    renderApp(["/"]);

    await waitFor(() => {
      expect(
        screen.getByTestId("first-run-onboarding-screen"),
      ).toBeInTheDocument();
    });
    expect(await screen.findByTestId("onboarding-modal")).toBeInTheDocument();
    expect(
      await screen.findByTestId("onboarding-step-check-backend"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("api-key-entry-screen"),
    ).not.toBeInTheDocument();
  });

  it("shows first-run onboarding before the recovery modal when no backend is configured", async () => {
    vi.stubEnv("VITE_SESSION_API_KEY", "");
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_SESSION_API_KEY__;
    window.localStorage.clear();
    __resetActiveStoreForTests();

    renderApp(["/"]);

    await waitFor(() => {
      expect(
        screen.getByTestId("first-run-onboarding-screen"),
      ).toBeInTheDocument();
    });
    expect(await screen.findByTestId("onboarding-modal")).toBeInTheDocument();
    expect(
      screen.queryByTestId("agent-server-onboarding-screen"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("manage-backends-modal"),
    ).not.toBeInTheDocument();
  });

  it("preserves onboarding preview on the root first-run path", async () => {
    window.localStorage.clear();
    __resetActiveStoreForTests();

    renderApp(["/?previewOnboardingStep=2"]);

    const modal = await screen.findByTestId("onboarding-modal");
    expect(modal).toHaveAttribute("data-initial-step", "2");
    expect(modal).toHaveAttribute("data-is-preview", "true");

    fireEvent.click(screen.getByTestId("mock-onboarding-launch"));

    expect(
      window.sessionStorage.getItem(
        `${ONBOARDING_DISMISSED_SESSION_KEY_PREFIX}:${NO_BACKEND_ID}`,
      ),
    ).toBeNull();
    expect(
      window.localStorage.getItem(ONBOARDING_COMPLETED_STORAGE_KEY),
    ).toBeNull();
  });

  it("lets root-level onboarding navigate to the launched conversation before closing", async () => {
    server.use(
      http.get("*/server_info", () =>
        HttpResponse.json({ uptime: 0, idle_time: 0, version: "1.28.1" }),
      ),
    );

    renderApp(["/"]);

    fireEvent.click(await screen.findByTestId("mock-onboarding-launch"));

    await waitFor(() => {
      expect(screen.getByTestId("conversation-outlet")).toBeInTheDocument();
    });
    expect(
      window.sessionStorage.getItem(
        `${ONBOARDING_DISMISSED_SESSION_KEY_PREFIX}:${SEEDED_DEFAULT_BACKEND_ID}`,
      ),
    ).toBe("1");
    expect(
      screen.queryByTestId("first-run-onboarding-screen"),
    ).not.toBeInTheDocument();
  });

  it("shows first-run onboarding before the recovery modal when locked to Cloud with no backend", async () => {
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://app.all-hands.dev");
    vi.stubEnv("VITE_SESSION_API_KEY", "");
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_SESSION_API_KEY__;
    window.localStorage.clear();
    __resetActiveStoreForTests();

    renderApp(["/"]);

    await waitFor(() => {
      expect(
        screen.getByTestId("first-run-onboarding-screen"),
      ).toBeInTheDocument();
    });
    expect(await screen.findByTestId("onboarding-modal")).toBeInTheDocument();
    expect(
      screen.queryByTestId("agent-server-onboarding-screen"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("manage-backends-modal"),
    ).not.toBeInTheDocument();
  });

  it("shows first-run onboarding when locked to Cloud even if a session API key is baked in", async () => {
    // Reproduces Hiep's report on PR #1389: a pre-built bundle with a baked-in
    // VITE_SESSION_API_KEY plus --lock-to-cloud used to seed a disconnected
    // Local backend, which skipped onboarding and landed on the Manage Backends
    // recovery modal. Locked mode must not seed a Local backend, so onboarding
    // still owns the first-run Cloud login.
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://app.all-hands.dev");
    vi.stubEnv("VITE_SESSION_API_KEY", "baked-session-key");
    (
      window as unknown as Record<string, unknown>
    ).__AGENT_CANVAS_SESSION_API_KEY__ = "baked-session-key";
    window.localStorage.clear();
    __resetActiveStoreForTests();

    renderApp(["/"]);

    await waitFor(() => {
      expect(
        screen.getByTestId("first-run-onboarding-screen"),
      ).toBeInTheDocument();
    });
    expect(await screen.findByTestId("onboarding-modal")).toBeInTheDocument();
    expect(
      screen.queryByTestId("agent-server-onboarding-screen"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("manage-backends-modal"),
    ).not.toBeInTheDocument();
    // No Local backend should have been seeded into the registry.
    expect(window.localStorage.getItem("openhands-backends")).toBeNull();
  });

  it("shows first-run onboarding when locked to Cloud with a stale persisted Local backend", async () => {
    // A Local backend persisted from a previous non-locked session must not
    // bypass onboarding once the deployment is locked to Cloud.
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://app.all-hands.dev");
    vi.stubEnv("VITE_SESSION_API_KEY", "");
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_SESSION_API_KEY__;
    window.localStorage.setItem(
      "openhands-backends",
      JSON.stringify([
        {
          id: "default-local",
          name: "Local",
          host: "http://127.0.0.1:8000",
          apiKey: "stale-key",
          kind: "local",
        },
      ]),
    );
    window.localStorage.setItem(
      "openhands-active-backend",
      JSON.stringify({ backendId: "default-local", orgId: null }),
    );
    __resetActiveStoreForTests();

    renderApp(["/"]);

    await waitFor(() => {
      expect(
        screen.getByTestId("first-run-onboarding-screen"),
      ).toBeInTheDocument();
    });
    expect(await screen.findByTestId("onboarding-modal")).toBeInTheDocument();
    expect(
      screen.queryByTestId("manage-backends-modal"),
    ).not.toBeInTheDocument();
  });

  it("forces first-run onboarding in locked mode even when a stale Local backend reports a configured LLM", async () => {
    // Critical regression for PR #1389 review: in locked-to-Cloud mode the
    // stale Local backend must not bypass onboarding, even when it happens
    // to report a configured LLM. The user must be routed through the Cloud
    // login / replacement flow instead.
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://app.all-hands.dev");
    vi.stubEnv("VITE_SESSION_API_KEY", "");
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_SESSION_API_KEY__;
    window.localStorage.setItem(
      "openhands-backends",
      JSON.stringify([
        {
          id: "user-added-local",
          name: "My agent-server",
          host: "http://127.0.0.1:8000",
          apiKey: "stale-key",
          kind: "local",
        },
      ]),
    );
    window.localStorage.setItem(
      "openhands-active-backend",
      JSON.stringify({ backendId: "user-added-local", orgId: null }),
    );
    __resetActiveStoreForTests();
    server.use(
      http.get("*/api/settings", () =>
        HttpResponse.json({
          llm_api_key_is_set: true,
          agent_settings: {
            llm: { model: "openai/gpt-5.5", api_key: "stored" },
          },
        }),
      ),
      http.get("*/server_info", () =>
        HttpResponse.json({ uptime: 0, idle_time: 0, version: "1.28.1" }),
      ),
    );

    renderApp(["/"]);

    await waitFor(() => {
      expect(
        screen.getByTestId("first-run-onboarding-screen"),
      ).toBeInTheDocument();
    });
    expect(await screen.findByTestId("onboarding-modal")).toBeInTheDocument();
    expect(
      screen.queryByTestId("manage-backends-modal"),
    ).not.toBeInTheDocument();
    // Backend readiness must NOT persist onboarding completion.
    expect(
      window.localStorage.getItem(ONBOARDING_COMPLETED_STORAGE_KEY),
    ).toBeNull();
  });

  it("forces first-run onboarding in locked mode when a Cloud backend points at a different host with a configured LLM", async () => {
    // Companion to the stale-Local test: a Cloud backend on a *different*
    // host than the locked Cloud host must also be forced through
    // onboarding, even if it reports a configured LLM. `kind === "cloud"`
    // alone is not enough — the host must match the locked host.
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://app.all-hands.dev");
    vi.stubEnv("VITE_SESSION_API_KEY", "");
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_SESSION_API_KEY__;
    const otherCloud = {
      id: "other-cloud",
      name: "Other Cloud",
      host: "https://other-cloud.example.com",
      apiKey: "other-token",
      kind: "cloud",
    };
    window.localStorage.setItem(
      "openhands-backends",
      JSON.stringify([otherCloud]),
    );
    window.localStorage.setItem(
      "openhands-active-backend",
      JSON.stringify({ backendId: otherCloud.id, orgId: null }),
    );
    __resetActiveStoreForTests();
    server.use(
      http.get("*/api/settings", () =>
        HttpResponse.json({
          llm_api_key_set: true,
          agent_settings: {
            llm: { model: "openai/gpt-5.5" },
          },
        }),
      ),
      http.get("*/server_info", () =>
        HttpResponse.json({ uptime: 0, idle_time: 0, version: "1.28.1" }),
      ),
    );

    renderApp(["/"]);

    await waitFor(() => {
      expect(
        screen.getByTestId("first-run-onboarding-screen"),
      ).toBeInTheDocument();
    });
    expect(await screen.findByTestId("onboarding-modal")).toBeInTheDocument();
    expect(
      window.localStorage.getItem(ONBOARDING_COMPLETED_STORAGE_KEY),
    ).toBeNull();
  });

  it("shows first-run onboarding when locked to Cloud even if onboarding was previously completed", async () => {
    // Reproduces hieptl's report on PR #1389: the user had previously
    // completed onboarding in a non-locked session (so the
    // `openhands-onboarded` localStorage flag is set), then relaunched the
    // static server with --lock-to-cloud. The stale completion flag used to
    // suppress first-run onboarding, so the app fell through to the Manage
    // Backends recovery modal ("Add Backend") instead of going straight to
    // Cloud login. In locked-to-Cloud mode the completion flag must not
    // bypass onboarding when the active backend is not a connected Cloud
    // backend.
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://app.all-hands.dev");
    vi.stubEnv("VITE_SESSION_API_KEY", "");
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_SESSION_API_KEY__;
    window.localStorage.clear();
    window.localStorage.setItem(ONBOARDING_COMPLETED_STORAGE_KEY, "1");
    __resetActiveStoreForTests();

    renderApp(["/"]);

    await waitFor(() => {
      expect(
        screen.getByTestId("first-run-onboarding-screen"),
      ).toBeInTheDocument();
    });
    expect(await screen.findByTestId("onboarding-modal")).toBeInTheDocument();
    expect(
      screen.queryByTestId("agent-server-onboarding-screen"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("manage-backends-modal"),
    ).not.toBeInTheDocument();
  });

  it("shows the auth gate after onboarding was skipped for the current session", async () => {
    vi.stubEnv("VITE_AUTH_REQUIRED", "true");
    vi.stubEnv("VITE_SESSION_API_KEY", "");
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_SESSION_API_KEY__;
    window.localStorage.clear();
    dismissOnboardingForBackend(NO_BACKEND_ID);
    __resetActiveStoreForTests();

    renderApp(["/"]);

    await waitFor(() => {
      expect(screen.getByTestId("api-key-entry-screen")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("onboarding-modal")).not.toBeInTheDocument();
  });

  it("shows the manage-backends modal when the connected server reports an old version", async () => {
    dismissOnboardingForBackend();
    server.use(
      http.get("/server_info", () =>
        HttpResponse.json({ uptime: 0, idle_time: 0, version: "1.27.1" }),
      ),
    );

    renderApp(["/"]);

    await waitFor(() => {
      expect(
        screen.getByTestId("agent-server-onboarding-screen"),
      ).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByTestId("manage-backends-modal")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("app-outlet")).not.toBeInTheDocument();
  });

  it("shows the manage-backends modal when the server omits a version field", async () => {
    dismissOnboardingForBackend();
    server.use(
      http.get("/server_info", () =>
        HttpResponse.json({ uptime: 0, idle_time: 0 }),
      ),
    );

    renderApp(["/"]);

    await waitFor(() => {
      expect(
        screen.getByTestId("agent-server-onboarding-screen"),
      ).toBeInTheDocument();
    });
    expect(screen.queryByTestId("app-outlet")).not.toBeInTheDocument();
  });

  it("shows the manage-backends modal when the backend is unreachable", async () => {
    let serverInfoRequests = 0;
    dismissOnboardingForBackend();

    // Use "*" prefix to match both relative paths and absolute URLs (e.g.,
    // http://127.0.0.1:8000/server_info) when VITE_BACKEND_BASE_URL is configured.
    server.use(
      http.get("*/server_info", () => {
        serverInfoRequests += 1;
        return HttpResponse.error();
      }),
    );

    renderApp(["/"]);

    await waitFor(() => {
      expect(
        screen.getByTestId("agent-server-onboarding-screen"),
      ).toBeInTheDocument();
    });

    // The onboarding placeholder now hosts the Manage Backends modal
    // directly so the user can edit/add a backend immediately. The
    // modal additionally probes /server_info per registered backend
    // for its status dot + version label, so the request count is
    // bounded but greater than the single config probe.
    await waitFor(() => {
      expect(screen.getByTestId("manage-backends-modal")).toBeInTheDocument();
    });
    expect(serverInfoRequests).toBeGreaterThanOrEqual(1);
    expect(screen.queryByTestId("app-outlet")).not.toBeInTheDocument();
  });

  it("shows the manage-backends recovery modal when the active cloud backend is logged out", async () => {
    const cloudBackend = {
      id: "cloud-expired",
      name: "OpenHands Cloud",
      host: "https://app.all-hands.dev",
      apiKey: "expired-token",
      kind: "cloud",
    };
    window.localStorage.setItem(
      "openhands-backends",
      JSON.stringify([cloudBackend]),
    );
    window.localStorage.setItem(
      "openhands-active-backend",
      JSON.stringify({ backendId: cloudBackend.id, orgId: null }),
    );
    window.sessionStorage.setItem(
      "openhands-active-backend",
      JSON.stringify({ backendId: cloudBackend.id, orgId: null }),
    );
    dismissOnboardingForBackend(cloudBackend.id);
    __resetActiveStoreForTests();
    server.use(
      http.get("https://app.all-hands.dev/api/keys/current", () =>
        HttpResponse.json({ detail: "NoCredentialsError" }, { status: 401 }),
      ),
    );

    renderApp(["/"]);

    await waitFor(() => {
      expect(
        screen.getByTestId("agent-server-onboarding-screen"),
      ).toBeInTheDocument();
    });
    expect(screen.getByTestId("manage-backends-modal")).toBeInTheDocument();
    expect(screen.getByText("Logged out")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Log back in" }),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("app-outlet")).not.toBeInTheDocument();
  });

  it("shows locked Cloud reconnect recovery without add-backend controls", async () => {
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://app.all-hands.dev");
    const cloudBackend = {
      id: "cloud-expired",
      name: "OpenHands Cloud",
      host: "https://app.all-hands.dev",
      apiKey: "expired-token",
      kind: "cloud",
    };
    window.localStorage.setItem(
      "openhands-backends",
      JSON.stringify([cloudBackend]),
    );
    window.localStorage.setItem(
      "openhands-active-backend",
      JSON.stringify({ backendId: cloudBackend.id, orgId: null }),
    );
    window.sessionStorage.setItem(
      "openhands-active-backend",
      JSON.stringify({ backendId: cloudBackend.id, orgId: null }),
    );
    window.localStorage.setItem(ONBOARDING_COMPLETED_STORAGE_KEY, "1");
    __resetActiveStoreForTests();
    server.use(
      http.get("https://app.all-hands.dev/api/keys/current", () =>
        HttpResponse.json({ detail: "NoCredentialsError" }, { status: 401 }),
      ),
    );

    renderApp(["/"]);

    await waitFor(() => {
      expect(
        screen.getByRole("heading", { name: "Reconnect to Cloud" }),
      ).toBeInTheDocument();
    });
    expect(screen.queryByTestId("manage-backends-add")).not.toBeInTheDocument();
    expect(
      screen.getByTestId("manage-backends-reconnect-cloud-login-button"),
    ).toHaveTextContent("Reconnect to Cloud");
    expect(screen.queryByTestId("app-outlet")).not.toBeInTheDocument();
  });

  it("renders the routed page when the agent server is reachable", async () => {
    dismissOnboardingForBackend();

    renderApp(["/"]);

    await waitFor(() => {
      expect(screen.getByTestId("app-outlet")).toBeInTheDocument();
    });

    expect(
      screen.queryByTestId("agent-server-onboarding-screen"),
    ).not.toBeInTheDocument();
  });

  it("renders the app for a configured shared local backend in a fresh browser profile", async () => {
    // A shared npm-installed agent-server keeps its LLM configuration across
    // browser sessions. A fresh browser has no `openhands-onboarded` marker,
    // but backend readiness must still suppress redundant onboarding.
    vi.stubEnv("VITE_BACKEND_BASE_URL", "http://127.0.0.1:8000");
    vi.stubEnv("VITE_SESSION_API_KEY", "test-session-key");
    // The launcher-seeded default-local backend (id
    // SEEDED_DEFAULT_BACKEND_ID) is created from these env stubs by
    // readStoredBackends().
    __resetActiveStoreForTests();
    server.use(
      http.get("*/api/settings", () =>
        HttpResponse.json({
          llm_api_key_is_set: true,
          agent_settings: {
            agent_kind: "openhands",
            llm: { model: "openai/gpt-5.5", api_key: "stored" },
          },
        }),
      ),
      http.get("*/api/profiles", () =>
        HttpResponse.json({
          profiles: [
            {
              name: "default",
              model: "openai/gpt-5.5",
              base_url: null,
              api_key_set: true,
            },
          ],
          active_profile: "default",
        }),
      ),
      http.get("*/api/agent-profiles", () =>
        HttpResponse.json({
          profiles: [
            {
              id: "agent-default",
              name: "default",
              agent_kind: "openhands",
              llm_profile_ref: "default",
            },
          ],
          active_agent_profile_id: "agent-default",
        }),
      ),
      http.get("*/server_info", () =>
        HttpResponse.json({ uptime: 0, idle_time: 0, version: "1.28.1" }),
      ),
    );

    renderApp(["/"]);

    await waitFor(() => {
      expect(screen.getByTestId("app-outlet")).toBeInTheDocument();
    });
    expect(
      screen.queryByTestId("first-run-onboarding-screen"),
    ).not.toBeInTheDocument();

    expect(
      window.localStorage.getItem(ONBOARDING_COMPLETED_STORAGE_KEY),
    ).toBeNull();
  });

  it("shows onboarding for an unconfigured backend despite a stale browser completion marker", async () => {
    window.localStorage.setItem(ONBOARDING_COMPLETED_STORAGE_KEY, "1");

    vi.stubEnv("VITE_BACKEND_BASE_URL", "http://127.0.0.1:8000");
    vi.stubEnv("VITE_SESSION_API_KEY", "test-session-key");
    __resetActiveStoreForTests();
    server.use(
      http.get("*/api/settings", () =>
        HttpResponse.json({
          agent_settings: {
            agent_kind: "openhands",
            llm: { model: "openai/gpt-5.5" },
          },
        }),
      ),
      http.get("*/api/profiles", () =>
        HttpResponse.json({
          profiles: [
            {
              name: "default",
              model: "openai/gpt-5.5",
              base_url: null,
              api_key_set: false,
            },
          ],
          active_profile: "default",
        }),
      ),
      http.get("*/api/profiles/default", () =>
        HttpResponse.json({
          name: "default",
          config: { model: "openai/gpt-5.5" },
          api_key_set: false,
        }),
      ),

      http.get("*/api/agent-profiles", () =>
        HttpResponse.json({
          profiles: [
            {
              id: "agent-default",
              name: "default",
              agent_kind: "openhands",
              llm_profile_ref: "default",
            },
          ],
          active_agent_profile_id: "agent-default",
        }),
      ),
      http.get("*/server_info", () =>
        HttpResponse.json({ uptime: 0, idle_time: 0, version: "1.28.1" }),
      ),
    );

    renderApp(["/"]);

    await waitFor(() => {
      expect(
        screen.getByTestId("first-run-onboarding-screen"),
      ).toBeInTheDocument();
    });
    expect(screen.queryByTestId("app-outlet")).not.toBeInTheDocument();
  });

  it("renders Cloud login directly for a fresh locked-to-Cloud first run", async () => {
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://app.all-hands.dev");
    vi.stubEnv("VITE_SESSION_API_KEY", "");
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_SESSION_API_KEY__;
    __resetActiveStoreForTests();

    renderApp(["/"]);

    await waitFor(() => {
      expect(
        screen.getByTestId("first-run-onboarding-screen"),
      ).toBeInTheDocument();
    });

    expect(await screen.findByTestId("onboarding-modal")).toBeInTheDocument();
    expect(screen.getByTestId("add-backend-cloud-title")).toBeVisible();
    expect(screen.getByTestId("add-backend-login-button")).toBeVisible();
    expect(
      screen.queryByTestId("onboarding-step-check-backend"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("onboarding-progress-bar"),
    ).not.toBeInTheDocument();
    expect(screen.queryByTestId("add-backend-close")).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("add-backend-advanced-toggle"),
    ).not.toBeInTheDocument();
    expect(screen.queryByTestId("app-outlet")).not.toBeInTheDocument();
  });

  it("redirects unauthenticated locked-cookie deployments to main app login", async () => {
    const assign = vi.fn();
    Object.defineProperty(window, "location", {
      configurable: true,
      value: {
        ...ORIGINAL_LOCATION,
        origin: "https://pr-254.staging.openhands.dev",
        hostname: "pr-254.staging.openhands.dev",
        pathname: "/canvas",
        search: "?tab=home",
        hash: "#top",
        assign,
      },
    });
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://pr-254.staging.all-hands.dev");
    vi.stubEnv("VITE_SESSION_API_KEY", "");
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_SESSION_API_KEY__;
    server.use(
      http.post("*/api/authenticate", () =>
        HttpResponse.json({ error: "unauthenticated" }, { status: 401 }),
      ),
    );
    __resetActiveStoreForTests();

    renderApp(["/"]);

    await waitFor(() => {
      expect(assign).toHaveBeenCalledWith(
        "/login?returnTo=%2Fcanvas%3Ftab%3Dhome%23top",
      );
    });
    expect(screen.queryByTestId("app-outlet")).not.toBeInTheDocument();
  });

  it("hides first-run onboarding immediately after Cloud login completes in locked-to-Cloud mode (no flicker)", async () => {
    // Regression for hieptl's flicker report on PR #1389: after Cloud
    // login succeeds in locked-to-Cloud mode, the onboarding modal's
    // onClose marks onboarding complete. The root first-run gate must
    // honor that completion IMMEDIATELY — without waiting for the Cloud
    // settings probe to confirm a configured LLM — so the first-run
    // screen disappears and the routed app renders, rather than the
    // modal flickering back via OnboardingHost. This test simulates the
    // post-login state (active locked Cloud backend + completion flag
    // set by the modal's onClose) with the Cloud settings probe
    // reporting NO configured LLM, which is exactly the window where
    // the old LLM-readiness gate kept the first-run screen mounted and
    // caused the reopen.
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://app.all-hands.dev");
    vi.stubEnv("VITE_SESSION_API_KEY", "");
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_SESSION_API_KEY__;
    const lockedCloud = {
      id: "locked-cloud",
      name: "OpenHands Cloud",
      host: "https://app.all-hands.dev",
      apiKey: "cloud-session-key",
      kind: "cloud",
    };
    window.localStorage.setItem(
      "openhands-backends",
      JSON.stringify([lockedCloud]),
    );
    window.localStorage.setItem(
      "openhands-active-backend",
      JSON.stringify({ backendId: lockedCloud.id, orgId: null }),
    );
    // The onboarding modal's onClose (markCompleted) sets this right
    // after Cloud login succeeds — before the Cloud settings probe
    // resolves. Seed it to reproduce the post-login moment.
    window.localStorage.setItem(ONBOARDING_COMPLETED_STORAGE_KEY, "1");
    __resetActiveStoreForTests();
    // Cloud settings probe reports no configured LLM. The completed
    // onboarding flag should still hide first-run onboarding once the
    // locked Cloud backend is active.
    server.use(
      http.get("https://app.all-hands.dev/api/v1/settings", () =>
        HttpResponse.json({ llm_api_key_set: false }),
      ),
      http.get("https://app.all-hands.dev/api/keys/current", () =>
        HttpResponse.json({ org_id: "org-1" }),
      ),
    );

    renderApp(["/"]);

    // The first-run onboarding screen must NOT be mounted (no reopen),
    // and the routed app must render instead.
    await waitFor(() => {
      expect(screen.getByTestId("app-outlet")).toBeInTheDocument();
    });
    expect(
      screen.queryByTestId("first-run-onboarding-screen"),
    ).not.toBeInTheDocument();
    expect(screen.queryByTestId("onboarding-modal")).not.toBeInTheDocument();
  });
});

describe("App root document links", () => {
  it("declares the SVG favicon used by the browser tab", () => {
    // Act
    const documentLinks = links();

    // Assert
    expect(documentLinks).toContainEqual({
      rel: "icon",
      type: "image/svg+xml",
      href: "/favicon.svg",
    });
  });

  it("prefixes document links when Canvas is mounted under a base path", () => {
    // Arrange
    vi.stubEnv("VITE_BASE_PATH", "/canvas");

    // Act
    const documentLinks = links();

    // Assert
    expect(documentLinks).toContainEqual({
      rel: "icon",
      type: "image/svg+xml",
      href: "/canvas/favicon.svg",
    });

    vi.unstubAllEnvs();
  });
});
