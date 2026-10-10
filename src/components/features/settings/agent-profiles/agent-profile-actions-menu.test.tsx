import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AgentProfileActionsMenu } from "./agent-profile-actions-menu";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("AgentProfileActionsMenu (#17426)", () => {
  const baseProps = {
    onEdit: vi.fn(),
    onSetActive: vi.fn(),
    onDelete: vi.fn(),
    onClose: vi.fn(),
    isActive: false,
    isActivating: false,
  };

  const renderMenu = (overrides: Partial<typeof baseProps> = {}) =>
    render(<AgentProfileActionsMenu {...baseProps} {...overrides} />);

  it("labels the activate action using the 'default' i18n key for consistency with the rest of the UI", () => {
    renderMenu();

    expect(screen.getByTestId("agent-profile-set-active")).toHaveTextContent(
      "SETTINGS$PROFILE_SET_DEFAULT",
    );
    expect(
      screen.queryByText("SETTINGS$PROFILE_SET_ACTIVE"),
    ).not.toBeInTheDocument();
  });

  it("invokes onSetActive when the activate menu item is clicked", () => {
    const onSetActive = vi.fn();
    renderMenu({ onSetActive });

    fireEvent.click(screen.getByTestId("agent-profile-set-active"));

    expect(onSetActive).toHaveBeenCalledTimes(1);
  });
});
