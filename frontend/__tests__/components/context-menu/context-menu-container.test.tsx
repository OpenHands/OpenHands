import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ContextMenuContainer } from "#/components/features/context-menu/context-menu-container";

describe("ContextMenuContainer", () => {
  const user = userEvent.setup();
  const onCloseMock = vi.fn();

  it("should render children", () => {
    render(
      <ContextMenuContainer onClose={onCloseMock}>
        <div data-testid="child-1">Child 1</div>
        <div data-testid="child-2">Child 2</div>
      </ContextMenuContainer>,
    );

    expect(screen.getByTestId("child-1")).toBeInTheDocument();
    expect(screen.getByTestId("child-2")).toBeInTheDocument();
  });

  it("should apply consistent base styling", () => {
    render(
      <ContextMenuContainer onClose={onCloseMock} testId="test-container">
        <div>Content</div>
      </ContextMenuContainer>,
    );

    const container = screen.getByTestId("test-container");
    expect(container).toHaveClass("bg-[#050505]");
    expect(container).toHaveClass("border");
    expect(container).toHaveClass("border-[#242424]");
    expect(container).toHaveClass("rounded-[12px]");
    expect(container).toHaveClass("p-[25px]");
    expect(container).toHaveClass("context-menu-box-shadow");
  });

  it("should call onClose when clicking outside", async () => {
    render(
      <ContextMenuContainer onClose={onCloseMock} testId="test-container">
        <div>Content</div>
      </ContextMenuContainer>,
    );

    await user.click(document.body);
    expect(onCloseMock).toHaveBeenCalledOnce();
  });

  it("should render children in a flex row layout", () => {
    render(
      <ContextMenuContainer onClose={onCloseMock} testId="test-container">
        <div data-testid="child-1">Child 1</div>
        <div data-testid="child-2">Child 2</div>
      </ContextMenuContainer>,
    );

    const container = screen.getByTestId("test-container");
    const innerDiv = container.firstChild as HTMLElement;
    expect(innerDiv).toHaveClass("flex");
    expect(innerDiv).toHaveClass("flex-row");
    expect(innerDiv).toHaveClass("gap-4");
  });

  it("should apply additional className when provided", () => {
    render(
      <ContextMenuContainer
        onClose={onCloseMock}
        testId="test-container"
        className="custom-class"
      >
        <div>Content</div>
      </ContextMenuContainer>,
    );

    const container = screen.getByTestId("test-container");
    expect(container).toHaveClass("custom-class");
  });
});
