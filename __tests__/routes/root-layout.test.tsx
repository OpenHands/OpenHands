import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { type ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createRoutesStub, data, Link } from "react-router";
import MainApp, { ErrorBoundary } from "#/routes/root-layout";
import { I18nKey } from "#/i18n/declaration";

const useConfigMock = vi.fn();
const useSettingsMock = vi.fn();
const changeLanguageMock = vi.fn();
const ensureActiveProfileMock = vi.fn();
const migrateEnabledSkillsMock = vi.fn();
const syncTelemetryConsentMock = vi.fn();
const syncAutomationTelemetryConsentMock = vi.fn();
const telemetryIdentityMock = vi.fn();

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => useConfigMock(),
}));

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => useSettingsMock(),
}));

vi.mock("#/hooks/use-migrate-enabled-skills", () => ({
  useMigrateEnabledSkills: () => migrateEnabledSkillsMock(),
}));

vi.mock("#/hooks/use-sync-telemetry-consent", () => ({
  useSyncTelemetryConsent: () => syncTelemetryConsentMock(),
}));

vi.mock("#/hooks/use-sync-automation-telemetry-consent", () => ({
  useSyncAutomationTelemetryConsent: () => syncAutomationTelemetryConsentMock(),
}));

vi.mock("#/hooks/use-telemetry-identity", () => ({
  useTelemetryIdentity: () => telemetryIdentityMock(),
}));

vi.mock("#/hooks/use-ensure-active-profile", () => ({
  useEnsureActiveProfile: () => ensureActiveProfileMock(),
}));

vi.mock(
  "#/components/features/canvas-extensions/canvas-extensions-runtime",
  () => ({
    CanvasExtensionsRuntimeProvider: ({ children }: { children: ReactNode }) =>
      children,
  }),
);

vi.mock("#/hooks/use-app-title", () => ({
  useAppTitle: () => "OpenHands",
}));

vi.mock("#/components/features/sidebar/sidebar", () => ({
  Sidebar: () => <div data-testid="sidebar" />,
}));

vi.mock("#/components/features/sidebar/sidebar-mobile-menu-bar", () => ({
  SidebarMobileMenuBar: () => <div data-testid="mobile-menu-bar" />,
}));

vi.mock("#/components/features/alerts/alert-banner", () => ({
  AlertBanner: ({
    errorMessage,
  }: {
    errorMessage: string | null | undefined;
  }) => <div data-testid="alert-banner">{errorMessage}</div>,
}));

vi.mock("#/components/features/backends/environment-switch-overlay", () => ({
  default: () => <div data-testid="environment-switch-overlay" />,
}));

vi.mock("#/components/features/command-menu/command-menu", () => ({
  CommandMenu: () => <div data-testid="command-menu" />,
}));

vi.mock("#/components/features/onboarding", () => ({
  OnboardingHost: () => <div data-testid="onboarding-host" />,
}));

vi.mock("react-i18next", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-i18next")>();
  return {
    ...actual,
    useTranslation: (namespace?: string) => ({
      t: (key: string) =>
        namespace === "openhands" ? key : `missing-namespace:${key}`,
    }),
  };
});

vi.mock("#/i18n", () => ({
  default: {
    changeLanguage: (...args: unknown[]) => changeLanguageMock(...args),
  },
}));

const RouterStub = createRoutesStub([
  {
    path: "/",
    Component: MainApp,
    children: [
      {
        path: "/",
        Component: () => (
          <>
            <div data-testid="outlet-content" />
            <Link to="/settings" data-testid="rerender-link">
              Settings
            </Link>
          </>
        ),
      },
      {
        path: "/automations",
        Component: () => <div data-testid="outlet-content" />,
      },
      {
        path: "/automations/:id",
        Component: () => <div data-testid="outlet-content" />,
      },
      {
        path: "/conversations",
        Component: () => <div data-testid="outlet-content" />,
      },
      {
        path: "/conversations/:id",
        Component: () => <div data-testid="outlet-content" />,
      },
      {
        path: "/settings",
        Component: () => <div data-testid="outlet-content" />,
      },
      {
        path: "*",
        Component: () => <div data-testid="outlet-content" />,
      },
    ],
  },
]);

function renderMainApp(path = "/") {
  const queryClient = new QueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <RouterStub initialEntries={[path]} />
    </QueryClientProvider>,
  );
}

function renderRouteError(thrown: unknown) {
  const ErrorRouterStub = createRoutesStub([
    {
      path: "/",
      Component: () => null,
      ErrorBoundary,
      loader: () => {
        throw thrown;
      },
    },
  ]);

  return render(<ErrorRouterStub initialEntries={["/"]} />);
}

describe("root layout", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useConfigMock.mockReturnValue({
      isLoading: false,
      data: {
        maintenance_start_time: null,
        faulty_models: [],
        error_message: null,
        updated_at: new Date().toISOString(),
      },
    });
    useSettingsMock.mockReturnValue({
      data: {
        language: "en",
        user_consents_to_analytics: true,
      },
    });
  });

  it("shows a loading spinner while config is loading", () => {
    useConfigMock.mockReturnValue({ isLoading: true, data: null });

    render(
      <QueryClientProvider client={new QueryClient()}>
        <RouterStub initialEntries={["/"]} />
      </QueryClientProvider>,
    );

    expect(screen.getByTestId("loading-spinner")).toBeInTheDocument();
  });

  it("does not render the analytics consent modal when analytics consent is missing", async () => {
    useSettingsMock.mockReturnValue({
      data: {
        language: "en",
        user_consents_to_analytics: null,
      },
    });

    renderMainApp();

    expect(screen.getByTestId("sidebar")).toBeInTheDocument();
    expect(screen.getByTestId("outlet-content")).toBeInTheDocument();
    // The analytics consent popup was removed from onboarding: a missing
    // (null) consent value must no longer surface the consent form.
    expect(
      screen.queryByTestId("user-capture-consent-form"),
    ).not.toBeInTheDocument();
    expect(await screen.findByTestId("command-menu")).toBeInTheDocument();
  });

  it("renders an identical root-layout className across routes so navigation never shifts the outer container", () => {
    const paths = [
      "/",
      "/automations/abc-123",
      "/conversations/abc-123",
      "/settings",
    ];

    const classNames = paths.map((path) => {
      const { unmount } = render(
        <QueryClientProvider client={new QueryClient()}>
          <RouterStub initialEntries={[path]} />
        </QueryClientProvider>,
      );
      const { className } = screen.getByTestId("root-layout");
      unmount();
      return className;
    });

    expect(new Set(classNames).size).toBe(1);
  });

  it("initializes layout services and renders the lazy global chrome", async () => {
    renderMainApp();

    expect(changeLanguageMock).toHaveBeenCalledWith("en");
    expect(migrateEnabledSkillsMock).toHaveBeenCalledOnce();
    expect(syncTelemetryConsentMock).toHaveBeenCalledOnce();
    expect(syncAutomationTelemetryConsentMock).toHaveBeenCalledOnce();
    expect(telemetryIdentityMock).toHaveBeenCalledOnce();
    expect(ensureActiveProfileMock).toHaveBeenCalledOnce();
    expect(document.title).toBe("OpenHands");
    expect(
      await screen.findByTestId("environment-switch-overlay"),
    ).toBeInTheDocument();
    expect(await screen.findByTestId("command-menu")).toBeInTheDocument();
    expect(screen.queryByTestId("alert-banner")).not.toBeInTheDocument();
  });

  it("updates the active language when the setting changes", () => {
    renderMainApp();
    expect(changeLanguageMock).toHaveBeenCalledWith("en");
    changeLanguageMock.mockClear();
    useSettingsMock.mockReturnValue({
      data: {
        language: "fr",
        user_consents_to_analytics: true,
      },
    });

    fireEvent.click(screen.getByTestId("rerender-link"));

    expect(changeLanguageMock).toHaveBeenCalledOnce();
    expect(changeLanguageMock).toHaveBeenCalledWith("fr");
  });

  it("leaves the current language unchanged when no language is configured", () => {
    useSettingsMock.mockReturnValue({
      data: {
        language: null,
        user_consents_to_analytics: true,
      },
    });

    renderMainApp();

    expect(changeLanguageMock).not.toHaveBeenCalled();
    expect(ensureActiveProfileMock).toHaveBeenCalledOnce();
  });

  it("renders while settings are loading without changing the language", () => {
    useSettingsMock.mockReturnValue({ data: undefined });

    renderMainApp();

    expect(screen.getByTestId("root-layout")).toBeInTheDocument();
    expect(changeLanguageMock).not.toHaveBeenCalled();
  });

  it("shows the mobile menu outside a conversation and hides it inside one", () => {
    const { unmount } = renderMainApp("/settings");

    expect(screen.getByTestId("mobile-menu-bar")).toBeInTheDocument();
    unmount();

    renderMainApp("/conversations/abc-123");

    expect(screen.queryByTestId("mobile-menu-bar")).not.toBeInTheDocument();
  });

  it("does not hide the mobile menu for a nested non-conversation route", () => {
    renderMainApp("/settings/conversations/abc-123");

    expect(screen.getByTestId("mobile-menu-bar")).toBeInTheDocument();
  });

  it("renders the onboarding host only when preview mode is requested", () => {
    const { unmount } = renderMainApp("/settings");

    expect(screen.queryByTestId("onboarding-host")).not.toBeInTheDocument();
    unmount();

    renderMainApp("/settings?previewOnboardingStep=2");

    expect(screen.getByTestId("onboarding-host")).toBeInTheDocument();
  });

  it("renders the lazy alert banner when the backend reports an error", async () => {
    useConfigMock.mockReturnValue({
      isLoading: false,
      data: {
        maintenance_start_time: null,
        faulty_models: [],
        error_message: "Backend degraded",
        updated_at: new Date().toISOString(),
      },
    });

    renderMainApp();

    expect(await screen.findByTestId("alert-banner")).toHaveTextContent(
      "Backend degraded",
    );
  });

  it.each([
    {
      label: "scheduled maintenance",
      data: {
        maintenance_start_time: "2030-01-02T03:04:05.000Z",
        faulty_models: [],
        error_message: null,
        updated_at: "2030-01-01T00:00:00.000Z",
      },
    },
    {
      label: "a faulty model",
      data: {
        maintenance_start_time: null,
        faulty_models: ["broken-model"],
        error_message: null,
        updated_at: "2030-01-01T00:00:00.000Z",
      },
    },
  ])("renders the alert banner for $label", async ({ data: configData }) => {
    useConfigMock.mockReturnValue({
      isLoading: false,
      data: configData,
    });

    renderMainApp();

    expect(await screen.findByTestId("alert-banner")).toBeInTheDocument();
  });

  it("renders without an alert when config data is unavailable", async () => {
    useConfigMock.mockReturnValue({ isLoading: false, data: null });

    renderMainApp();

    expect(await screen.findByTestId("command-menu")).toBeInTheDocument();
    expect(screen.queryByTestId("alert-banner")).not.toBeInTheDocument();
  });
});

describe("root layout error boundary", () => {
  it("renders an object route response with its HTTP details", async () => {
    renderRouteError(
      data(
        { reason: "maintenance" },
        { status: 503, statusText: "Service Unavailable" },
      ),
    );

    expect(
      await screen.findByRole("heading", { name: "503" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Service Unavailable")).toBeInTheDocument();
    expect(screen.getByText('{"reason":"maintenance"}')).toBeInTheDocument();
  });

  it("renders primitive route-response data without serializing it", async () => {
    renderRouteError(
      data("plain failure", { status: 400, statusText: "Bad Request" }),
    );

    expect(
      await screen.findByRole("heading", { name: "400" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Bad Request")).toBeInTheDocument();
    expect(screen.getByText("plain failure")).toBeInTheDocument();
  });

  it("renders the generic translated heading and message for an Error", async () => {
    renderRouteError(new Error("Kaboom"));

    expect(
      await screen.findByRole("heading", { name: I18nKey.ERROR$GENERIC }),
    ).toBeInTheDocument();
    expect(screen.getByText("Kaboom")).toBeInTheDocument();
  });

  it("renders the unknown translated heading for a non-Error value", async () => {
    renderRouteError("mystery failure");

    expect(
      await screen.findByRole("heading", { name: I18nKey.ERROR$UNKNOWN }),
    ).toBeInTheDocument();
  });
});
