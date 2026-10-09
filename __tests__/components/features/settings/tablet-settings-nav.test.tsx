import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { NavigationProvider } from "#/context/navigation-context";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { TabletSettingsNav } from "#/components/features/settings/tablet-settings-nav";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";

vi.mock("react-i18next", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-i18next")>()),
  useTranslation: () => ({
    t: (key: string, params?: Record<string, unknown>) => {
      const map: Record<string, string> = {
        SETTINGS$NAV_AGENT: "Agent",
        SETTINGS$LLM_PROFILES: "LLM Profiles",
        SETTINGS$APPLICATION: "Application",
        SETTINGS$INTEGRATIONS_SETTINGS_LINK: "Integrations",
        SETTINGS$CLOUD_SETTINGS_LINK: "All Cloud Settings",
        NAV$SECTION_NAV_LABEL: `Section navigation: ${params?.section ?? ""}`,
      };
      return map[key] ?? key;
    },
    i18n: { language: "en", exists: () => false },
  }),
}));

const mockNavigate = vi.fn();

function renderTabletSettingsNav(currentPath = "/settings/agents") {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>
        <NavigationProvider
          value={{
            currentPath,
            conversationId: null,
            isNavigating: false,
            navigate: mockNavigate,
          }}
        >
          <TabletSettingsNav />
        </NavigationProvider>
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

describe("TabletSettingsNav", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    __resetActiveStoreForTests();
  });

  it("does not render when on the /settings hub page", () => {
    renderTabletSettingsNav("/settings");
    expect(screen.queryByTestId("tablet-sibling-nav")).not.toBeInTheDocument();
  });

  it("renders on /settings/agents with current section label and excludes it from menu", async () => {
    const user = userEvent.setup();
    renderTabletSettingsNav("/settings/agents");

    const trigger = screen.getByTestId("tablet-sibling-nav-trigger");
    expect(trigger).toHaveTextContent("Agent");

    await user.click(trigger);
    const menu = screen.getByTestId("tablet-sibling-nav-menu");
    expect(menu).toBeInTheDocument();

    // /settings/agents is the current page, so it should be excluded
    expect(
      screen.queryByTestId("tablet-sibling-nav-item-/settings/agents"),
    ).not.toBeInTheDocument();

    // Sibling pages should be present
    expect(
      screen.getByTestId("tablet-sibling-nav-item-/settings/llm"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("tablet-sibling-nav-item-/settings/app"),
    ).toBeInTheDocument();
  });

  it("navigates to sibling destination when clicked", async () => {
    const user = userEvent.setup();
    renderTabletSettingsNav("/settings/agents");

    await user.click(screen.getByTestId("tablet-sibling-nav-trigger"));
    const appItem = screen.getByTestId("tablet-sibling-nav-item-/settings/app");
    await user.click(appItem);

    expect(mockNavigate).toHaveBeenCalledWith("/settings/app");
  });

  it("includes Cloud links when on a Cloud backend", async () => {
    const cloudBackend: Backend = {
      id: "cloud-1",
      name: "OpenHands Cloud",
      host: "https://app.all-hands.dev",
      apiKey: "test-token",
      kind: "cloud",
    };
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-1" });

    const user = userEvent.setup();
    renderTabletSettingsNav("/settings/app");

    await user.click(screen.getByTestId("tablet-sibling-nav-trigger"));
    expect(
      screen.getByTestId(
        "tablet-sibling-nav-item-https://app.all-hands.dev/settings/integrations?org=org-1",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId(
        "tablet-sibling-nav-item-https://app.all-hands.dev/settings?org=org-1",
      ),
    ).toBeInTheDocument();
  });
});
