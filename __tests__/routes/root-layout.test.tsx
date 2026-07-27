import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createRoutesStub } from "react-router";
import MainApp from "#/routes/root-layout";

const useConfigMock = vi.fn();
const useSettingsMock = vi.fn();

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => useConfigMock(),
}));

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => useSettingsMock(),
}));

vi.mock("#/hooks/use-sync-telemetry-consent", () => ({
  useSyncTelemetryConsent: () => {},
}));

vi.mock("#/hooks/use-telemetry-identity", () => ({
  useTelemetryIdentity: () => {},
}));

vi.mock("#/hooks/use-app-title", () => ({
  useAppTitle: () => "OpenHands",
}));

vi.mock("#/components/features/sidebar/sidebar", () => ({
  Sidebar: () => <div data-testid="sidebar" />,
}));

vi.mock("#/components/features/alerts/alert-banner", () => ({
  AlertBanner: () => <div data-testid="alert-banner" />,
}));

vi.mock("#/i18n", () => ({
  default: {
    changeLanguage: vi.fn(),
  },
}));

const RouterStub = createRoutesStub([
  {
    path: "/",
    Component: MainApp,
    children: [
      {
        path: "/",
        Component: () => <div data-testid="outlet-content" />,
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
    ],
  },
]);

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

  it("does not render the analytics consent modal when analytics consent is missing", () => {
    useSettingsMock.mockReturnValue({
      data: {
        language: "en",
        user_consents_to_analytics: null,
      },
    });

    render(
      <QueryClientProvider client={new QueryClient()}>
        <RouterStub initialEntries={["/"]} />
      </QueryClientProvider>,
    );

    expect(screen.getByTestId("sidebar")).toBeInTheDocument();
    expect(screen.getByTestId("outlet-content")).toBeInTheDocument();
    // The analytics consent popup was removed from onboarding: a missing
    // (null) consent value must no longer surface the consent form.
    expect(
      screen.queryByTestId("user-capture-consent-form"),
    ).not.toBeInTheDocument();
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
});
