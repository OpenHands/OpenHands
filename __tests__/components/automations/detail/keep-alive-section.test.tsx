import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import AutomationService from "#/api/automation-service/automation-service.api";
import { KeepAliveSection } from "#/components/features/automations/detail/keep-alive-section";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";

const automation: Automation = {
  id: "auto-1",
  name: "Comment cleanup",
  prompt: null,
  entrypoint: "python3 scripts/main.py",
  trigger: { type: "cron", schedule: "0 9 * * 1" },
  enabled: true,
  keep_alive: false,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

const continueConversationAutomation: Automation = {
  ...automation,
  trigger: {
    type: "event",
    source: "github",
    on: "issue_comment.created",
    destination: "continue_conversation",
  },
  keep_alive: true,
};

const expandAdvanced = (user: ReturnType<typeof userEvent.setup>) =>
  user.click(screen.getByTestId("automation-advanced-toggle"));

const getSwitch = () =>
  screen.getByRole("switch", { name: I18nKey.AUTOMATIONS$DETAIL$KEEP_ALIVE });

afterEach(() => {
  vi.restoreAllMocks();
});

describe("KeepAliveSection", () => {
  it("renders nothing for an automation service that does not return keep_alive", () => {
    // Arrange & Act
    const { container } = renderWithProviders(
      <KeepAliveSection
        automation={{ ...automation, keep_alive: undefined }}
        canEdit
      />,
    );

    // Assert
    expect(container).toBeEmptyDOMElement();
  });

  it("keeps the setting collapsed under Advanced until expanded", async () => {
    // Arrange
    const user = userEvent.setup();
    renderWithProviders(<KeepAliveSection automation={automation} canEdit />);
    const toggle = screen.getByTestId("automation-advanced-toggle");
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("switch")).not.toBeInTheDocument();

    // Act
    await expandAdvanced(user);

    // Assert
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(getSwitch()).toBeInTheDocument();
  });

  it.each([
    { saved: false, next: true },
    { saved: true, next: false },
  ])(
    "shows the saved value $saved and saves $next with one PATCH",
    async ({ saved, next }) => {
      // Arrange
      const user = userEvent.setup();
      const update = vi
        .spyOn(AutomationService, "updateAutomation")
        .mockResolvedValue({ ...automation, keep_alive: next });
      renderWithProviders(
        <KeepAliveSection
          automation={{ ...automation, keep_alive: saved }}
          canEdit
        />,
      );
      await expandAdvanced(user);
      expect(getSwitch()).toHaveAttribute("aria-checked", String(saved));
      expect(
        screen.getByText(I18nKey.AUTOMATIONS$DETAIL$KEEP_ALIVE_DESCRIPTION),
      ).toBeInTheDocument();

      // Act
      await user.click(getSwitch());

      // Assert
      expect(update).toHaveBeenCalledTimes(1);
      expect(update).toHaveBeenCalledWith("auto-1", { keep_alive: next });
      await waitFor(() => expect(getSwitch()).toBeEnabled());
    },
  );

  it("treats a null keep_alive as off", async () => {
    // Arrange
    const user = userEvent.setup();
    renderWithProviders(
      <KeepAliveSection
        automation={{ ...automation, keep_alive: null }}
        canEdit
      />,
    );

    // Act
    await expandAdvanced(user);

    // Assert
    expect(getSwitch()).toHaveAttribute("aria-checked", "false");
  });

  it("is read-only for someone who cannot edit the automation", async () => {
    // Arrange
    const user = userEvent.setup();
    const update = vi.spyOn(AutomationService, "updateAutomation");
    renderWithProviders(
      <KeepAliveSection automation={automation} canEdit={false} />,
    );
    await expandAdvanced(user);

    // Act
    await user.click(getSwitch());

    // Assert
    expect(getSwitch()).toBeDisabled();
    expect(update).not.toHaveBeenCalled();
  });

  it("shows keep_alive as forced on for a continue_conversation trigger", async () => {
    // Arrange
    const user = userEvent.setup();
    const update = vi.spyOn(AutomationService, "updateAutomation");
    renderWithProviders(
      <KeepAliveSection automation={continueConversationAutomation} canEdit />,
    );
    await expandAdvanced(user);

    // Act
    await user.click(getSwitch());

    // Assert
    expect(getSwitch()).toHaveAttribute("aria-checked", "true");
    expect(getSwitch()).toBeDisabled();
    expect(
      screen.getByText(I18nKey.AUTOMATIONS$DETAIL$KEEP_ALIVE_FORCED),
    ).toBeInTheDocument();
    expect(update).not.toHaveBeenCalled();
  });
});
