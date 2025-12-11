import { afterEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { createRoutesStub } from "react-router";
import { waitFor } from "@testing-library/react";
import { Sidebar } from "#/components/features/sidebar/sidebar";
import SettingsService from "#/api/settings-service/settings-service.api";

// These tests will now fail because the conversation panel is rendered through a portal
// and technically not a child of the Sidebar component.

const RouterStub = createRoutesStub([
  {
    path: "/conversation/:conversationId",
    Component: () => <Sidebar />,
  },
]);

const renderSidebar = () =>
  renderWithProviders(<RouterStub initialEntries={["/conversation/123"]} />);

describe("Sidebar", () => {
  const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should fetch settings data on mount", async () => {
    renderSidebar();
    await waitFor(() => expect(getSettingsSpy).toHaveBeenCalled());
  });
});
