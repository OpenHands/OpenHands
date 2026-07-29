import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ContextMenuListItem } from "#/components/features/context-menu/context-menu-list-item";

describe("ContextMenuListItem", () => {
  it("should render the component with the children", () => {
    const onClickMock = vi.fn();
    render(
      <ContextMenuListItem onClick={onClickMock}>Test</ContextMenuListItem>,
    );

    expect(screen.getByTestId("context-menu-list-item")).toBeInTheDocument();
    expect(screen.getByText("Test")).toBeInTheDocument();
  });

  it("should call the onClick callback when clicked", async () => {
    const user = userEvent.setup();
    const onClickMock = vi.fn();
    render(
      <ContextMenuListItem onClick={onClickMock}>Test</ContextMenuListItem>,
    );

    const element = screen.getByTestId("context-menu-list-item");
    await user.click(element);

    expect(onClickMock).toHaveBeenCalledOnce();
  });

  it("should not call the onClick callback when clicked and the button is disabled", async () => {
    const user = userEvent.setup();
    const onClickMock = vi.fn();
    render(
      <ContextMenuListItem onClick={onClickMock} isDisabled>
        Test
      </ContextMenuListItem>,
    );

    const element = screen.getByTestId("context-menu-list-item");
    await user.click(element);

    expect(onClickMock).not.toHaveBeenCalled();
  });
});
