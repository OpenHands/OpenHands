import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createRoutesStub } from "react-router";
import SettingsIndex from "#/routes/settings-index";
import OptionService from "#/api/option-service/option-service.api";
import { __resetActiveStoreForTests } from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";

function renderSettingsIndex() {
  const RouterStub = createRoutesStub([
    { path: "/settings", Component: SettingsIndex },
    {
      path: "/settings/agents",
      Component: () => <div data-testid="agent-profiles-screen" />,
    },
    {
      path: "/settings/app",
      Component: () => <div data-testid="app-settings-screen" />,
    },
  ]);

  return render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <ActiveBackendProvider>
        <RouterStub initialEntries={["/settings"]} />
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

describe("settings index route", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
    __resetActiveStoreForTests();
    vi.spyOn(OptionService, "getConfig").mockResolvedValue({
      feature_flags: {
        hide_llm_settings: false,
        hide_users_page: true,
      },
      providers_configured: [],
      maintenance_start_time: null,
      recaptcha_site_key: null,
      faulty_models: [],
      error_message: null,
      updated_at: new Date().toISOString(),
    });
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    window.localStorage.clear();
    __resetActiveStoreForTests();
  });

  it("lands on the Agent library by default", async () => {
    // Arrange — no lock-to-cloud; jsdom's default viewport is desktop width.
    // Act
    renderSettingsIndex();

    // Assert
    expect(
      await screen.findByTestId("agent-profiles-screen"),
    ).toBeInTheDocument();
  });

  it("lands on Application when the canvas is locked to a Cloud host", async () => {
    // Arrange — SaaS / self-hosted OHE serve the canvas with `--lock-to-cloud`.
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://app.all-hands.dev");

    // Act
    renderSettingsIndex();

    // Assert
    expect(
      await screen.findByTestId("app-settings-screen"),
    ).toBeInTheDocument();
  });
});
