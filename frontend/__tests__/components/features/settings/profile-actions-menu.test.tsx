import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ProfileActionsMenu } from "#/components/features/settings/profile-actions-menu";

function renderMenu(
  overrides: Partial<React.ComponentProps<typeof ProfileActionsMenu>> = {},
) {
  const props = {
    onEdit: vi.fn(),
    onRename: vi.fn(),
    onSetActive: vi.fn(),
    onDelete: vi.fn(),
    onClose: vi.fn(),
    isActive: false,
    isActivating: false,
    ...overrides,
  };
  return {
    // eslint-disable-next-line react/jsx-props-no-spreading
    ...render(<ProfileActionsMenu {...props} />),
    props,
  };
}

describe("ProfileActionsMenu", () => {
  it("renders edit, rename, set-active, and delete items", () => {
    renderMenu();

    expect(screen.getByTestId("profile-edit")).toBeInTheDocument();
    expect(screen.getByTestId("profile-rename")).toBeInTheDocument();
    expect(screen.getByTestId("profile-set-active")).toBeInTheDocument();
    expect(screen.getByTestId("profile-delete")).toBeInTheDocument();
  });

  it("invokes the matching callback then onClose when an item is clicked", async () => {
    const { props } = renderMenu();
    const user = userEvent.setup();

    await user.click(screen.getByTestId("profile-edit"));
    expect(props.onEdit).toHaveBeenCalledTimes(1);
    expect(props.onClose).toHaveBeenCalledTimes(1);
  });

  it("disables Set as active when the profile is already active", () => {
    renderMenu({ isActive: true });

    expect(screen.getByTestId("profile-set-active")).toBeDisabled();
  });

  it("disables Set as active while an activation is in flight", () => {
    renderMenu({ isActivating: true });

    expect(screen.getByTestId("profile-set-active")).toBeDisabled();
  });
});
