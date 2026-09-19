import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { DetailHeader } from "#/components/features/automations/detail/detail-header";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";

const disabledAutomation: Automation = {
  id: "auto-1",
  name: "Daily digest",
  prompt: "Summarize",
  trigger: {
    type: "cron",
    schedule: "0 9 * * *",
    schedule_human: "Daily at 09:00",
  },
  enabled: false,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

function renderHeader(
  props: Partial<React.ComponentProps<typeof DetailHeader>> = {},
) {
  return render(
    <DetailHeader
      automation={disabledAutomation}
      onToggle={vi.fn()}
      onDelete={vi.fn()}
      onExport={vi.fn()}
      onDownloadTarball={vi.fn()}
      {...props}
    />,
  );
}

describe("DetailHeader — enabled toggle gating", () => {
  it("hides the switch and Turn on when canToggle is false but keeps Delete", async () => {
    // Arrange — a manager who did not create this disabled automation.
    const user = userEvent.setup();
    renderHeader({ canManage: true, canToggle: false });

    // Act
    await user.click(screen.getByLabelText(I18nKey.AUTOMATIONS$ACTIONS_MENU));

    // Assert
    expect(screen.queryByRole("switch")).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: I18nKey.AUTOMATIONS$TURN_ON }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: I18nKey.AUTOMATIONS$DELETE }),
    ).toBeInTheDocument();
  });

  it("follows canManage for the switch when canToggle is omitted", () => {
    // Arrange / Act
    renderHeader({ canManage: true });

    // Assert
    expect(
      screen.getByRole("switch", { name: I18nKey.AUTOMATIONS$TURN_ON }),
    ).toBeInTheDocument();
  });
});
