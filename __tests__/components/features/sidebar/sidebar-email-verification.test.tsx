import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Sidebar } from "#/components/features/sidebar/sidebar";
import {
  NavigationProvider,
  type NavigationContextValue,
} from "#/context/navigation-context";
import { SidebarMobileNavProvider } from "#/components/features/sidebar/sidebar-mobile-nav-context";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { useSidebarStore } from "#/stores/sidebar-store";
import { useSettings } from "#/hooks/query/use-settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import OptionService from "#/api/option-service/option-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import type { Settings } from "#/types/settings";
import type { WebClientConfig } from "#/api/option-service/option.types";
import type { Backend } from "#/api/backend-registry/types";

// Isolate the sidebar down to its top-level navigation. These are peripheral to
// email-verification gating: the conversation list and backend selector fetch
// their own data, the command menu is a search launcher, and the logo is
// decorative. Mocking the *components* (not the settings/backend hooks under
// test) keeps the render deterministic.
vi.mock("#/components/features/sidebar/sidebar-conversation-list", () => ({
  SidebarConversationList: () => (
    <div data-testid="sidebar-conversation-list" />
  ),
}));
vi.mock("#/components/features/backends/backend-selector", () => ({
  BackendSelector: () => <div data-testid="backend-selector" />,
}));
vi.mock("#/components/features/command-menu/command-menu-trigger", () => ({
  CommandMenuTrigger: () => <div data-testid="command-menu-trigger" />,
}));
vi.mock("#/components/shared/buttons/openhands-logo-button", () => ({
  OpenHandsLogoButton: () => <div data-testid="logo-button" />,
}));

// The backend-health hook polls each backend over the network to drive a status
// dot shown only in the collapsed rail — unrelated to the behavior under test.
// Stub it so no probe fires (service-level mocking isn't practical here: the
// probe instantiates SDK clients inline).
vi.mock("#/hooks/query/use-backends-health", () => ({
  useBackendsHealth: () => ({}),
}));

const TEST_BACKEND: Backend = {
  id: "test-local",
  name: "Local",
  host: "http://localhost:3000",
  apiKey: "test-key",
  kind: "local",
};

function buildSettings(overrides: Partial<Settings> = {}): Settings {
  return {
    ...MOCK_DEFAULT_USER_SETTINGS,
    ...overrides,
    agent_settings:
      overrides.agent_settings ?? MOCK_DEFAULT_USER_SETTINGS.agent_settings,
  };
}

function buildConfig(): WebClientConfig {
  return {
    feature_flags: { hide_llm_settings: false, hide_users_page: true },
    providers_configured: [],
    maintenance_start_time: null,
    recaptcha_site_key: null,
    faulty_models: [],
    error_message: null,
    updated_at: "2026-01-01T00:00:00.000Z",
  };
}

// Renders the sidebar next to a probe that flips a marker once the settings
// query settles with an unverified email. Waiting on the marker means the
// assertions observe the resolved `email_verified: false` state — the exact
// condition that used to disable the links — rather than the loading frame.
function ProbedSidebar() {
  const settings = useSettings();
  return (
    <>
      <Sidebar />
      {settings.isFetched && settings.data?.email_verified === false ? (
        <div data-testid="settings-loaded" />
      ) : null}
    </>
  );
}

function renderSidebar() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const navigation: NavigationContextValue = {
    currentPath: "/",
    conversationId: null,
    isNavigating: false,
    navigate: vi.fn(),
  };
  return render(
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>
        <NavigationProvider value={navigation}>
          <SidebarMobileNavProvider>
            <ProbedSidebar />
          </SidebarMobileNavProvider>
        </NavigationProvider>
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

describe("Sidebar email verification", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
    window.sessionStorage.clear();
    useSidebarStore.setState({ collapsed: false });
    // A resolved local backend is required; otherwise `useSettings`
    // short-circuits to defaults and never calls the mocked service.
    setRegisteredBackends([TEST_BACKEND]);
    setActiveSelection({ backendId: TEST_BACKEND.id, orgId: null });
    vi.spyOn(OptionService, "getConfig").mockResolvedValue(buildConfig());
  });

  afterEach(() => {
    window.localStorage.clear();
    window.sessionStorage.clear();
  });

  it("keeps primary navigation enabled when the user's email is unverified", async () => {
    // Arrange: the OAuth-changed-email state — the backend stays permissive but
    // reports the address as unverified.
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({ email_verified: false }),
    );

    // Act
    renderSidebar();
    await screen.findByTestId("settings-loaded");

    // Assert: New Chat, Customize/Extensions, and Automations stay actionable
    // (a disabled link would carry aria-disabled="true").
    for (const testId of [
      "sidebar-conversations-link",
      "sidebar-skills-link",
      "sidebar-automations-link",
    ]) {
      expect(screen.getByTestId(testId)).not.toHaveAttribute("aria-disabled");
    }
  });
});
