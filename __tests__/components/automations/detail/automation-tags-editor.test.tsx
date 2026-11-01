import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import AutomationService from "#/api/automation-service/automation-service.api";
import { AutomationTagsEditor } from "#/components/features/automations/detail/automation-tags-editor";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";

const automation: Automation = {
  id: "auto-1",
  name: "PR reviewer",
  prompt: "Review",
  trigger: { type: "cron", schedule: "0 9 * * *" },
  enabled: true,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  tags: ["ci"],
};

afterEach(() => {
  vi.restoreAllMocks();
});

describe("AutomationTagsEditor", () => {
  it("shows tags read-only to someone other than the creator", () => {
    // Arrange & Act
    renderWithProviders(
      <AutomationTagsEditor automation={automation} canEdit={false} />,
    );

    // Assert
    expect(screen.getByTestId("automation-tag")).toHaveTextContent("ci");
    expect(
      screen.queryByTestId("automation-tags-edit"),
    ).not.toBeInTheDocument();
  });

  it("renders nothing for an automation service without tag support", () => {
    // Arrange & Act
    const { container } = renderWithProviders(
      <AutomationTagsEditor
        automation={{ ...automation, tags: undefined }}
        canEdit
      />,
    );

    // Assert
    expect(container).toBeEmptyDOMElement();
  });

  it("saves the creator's edited list in one update, normalized", async () => {
    // Arrange
    const user = userEvent.setup();
    const update = vi
      .spyOn(AutomationService, "updateAutomation")
      .mockResolvedValue({ ...automation, tags: ["software-factory"] });
    renderWithProviders(
      <AutomationTagsEditor automation={automation} canEdit />,
    );

    // Act — add a tag typed the way people write it, then drop the old one.
    await user.click(screen.getByTestId("automation-tags-edit"));
    await user.type(
      screen.getByTestId("automation-tags-input"),
      "Software Factory{Enter}",
    );
    const ciPill = screen
      .getAllByTestId("automation-tag")
      .find((pill) => pill.textContent === "ci")!;
    await user.click(
      within(ciPill).getByLabelText(I18nKey.AUTOMATIONS$REMOVE_TAG),
    );
    await user.click(screen.getByTestId("automation-tags-save"));

    // Assert
    expect(update).toHaveBeenCalledWith("auto-1", {
      tags: ["software-factory"],
    });
    await waitFor(() =>
      expect(
        screen.queryByTestId("automation-tags-editor"),
      ).not.toBeInTheDocument(),
    );
  });

  it("rejects a tag the service would refuse without saving", async () => {
    // Arrange
    const user = userEvent.setup();
    const update = vi.spyOn(AutomationService, "updateAutomation");
    renderWithProviders(
      <AutomationTagsEditor automation={automation} canEdit />,
    );

    // Act
    await user.click(screen.getByTestId("automation-tags-edit"));
    await user.type(screen.getByTestId("automation-tags-input"), "q3_release");
    await user.click(screen.getByTestId("automation-tags-save"));

    // Assert
    expect(screen.getByRole("alert")).toHaveTextContent(
      I18nKey.AUTOMATIONS$TAG_ERROR_INVALID,
    );
    expect(update).not.toHaveBeenCalled();
  });
});
