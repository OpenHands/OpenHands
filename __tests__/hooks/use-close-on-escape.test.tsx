import React from "react";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useCloseOnEscape } from "#/hooks/use-close-on-escape";

function Layer({
  name,
  isOpen,
  onClose,
}: {
  name: string;
  isOpen: boolean;
  onClose: () => void;
}) {
  const triggerRef = React.useRef<HTMLButtonElement>(null);
  useCloseOnEscape(isOpen, onClose, triggerRef);
  return (
    <button ref={triggerRef} type="button" data-testid={`${name}-trigger`}>
      {name}
    </button>
  );
}

function pressEscape() {
  fireEvent.keyDown(document.body, { key: "Escape" });
}

describe("useCloseOnEscape", () => {
  it("closes only the most recently opened layer, then the one below", () => {
    const closeFirst = vi.fn();
    const closeSecond = vi.fn();
    const { rerender } = render(
      <>
        <Layer name="first" isOpen onClose={closeFirst} />
        <Layer name="second" isOpen={false} onClose={closeSecond} />
      </>,
    );
    // Open the second layer after the first, as a user would.
    rerender(
      <>
        <Layer name="first" isOpen onClose={closeFirst} />
        <Layer name="second" isOpen onClose={closeSecond} />
      </>,
    );

    pressEscape();
    expect(closeSecond).toHaveBeenCalledTimes(1);
    expect(closeFirst).not.toHaveBeenCalled();
    expect(screen.getByTestId("second-trigger")).toHaveFocus();

    rerender(
      <>
        <Layer name="first" isOpen onClose={closeFirst} />
        <Layer name="second" isOpen={false} onClose={closeSecond} />
      </>,
    );
    pressEscape();
    expect(closeFirst).toHaveBeenCalledTimes(1);
    expect(closeSecond).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("first-trigger")).toHaveFocus();
  });

  it("orders layers by when they opened, not by render order", () => {
    const closeFirst = vi.fn();
    const closeSecond = vi.fn();
    const { rerender } = render(
      <>
        <Layer name="first" isOpen={false} onClose={closeFirst} />
        <Layer name="second" isOpen onClose={closeSecond} />
      </>,
    );
    rerender(
      <>
        <Layer name="first" isOpen onClose={closeFirst} />
        <Layer name="second" isOpen onClose={closeSecond} />
      </>,
    );

    pressEscape();
    expect(closeFirst).toHaveBeenCalledTimes(1);
    expect(closeSecond).not.toHaveBeenCalled();
  });

  it("ignores an Escape that something inside the layer already handled", () => {
    const onClose = vi.fn();
    render(<Layer name="only" isOpen onClose={onClose} />);

    const handled = new KeyboardEvent("keydown", {
      key: "Escape",
      bubbles: true,
      cancelable: true,
    });
    handled.preventDefault();
    act(() => {
      document.body.dispatchEvent(handled);
    });
    expect(onClose).not.toHaveBeenCalled();

    pressEscape();
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("stops listening once every layer is closed or unmounted", () => {
    const onClose = vi.fn();
    const { rerender, unmount } = render(
      <Layer name="only" isOpen onClose={onClose} />,
    );
    rerender(<Layer name="only" isOpen={false} onClose={onClose} />);
    pressEscape();
    expect(onClose).not.toHaveBeenCalled();

    rerender(<Layer name="only" isOpen onClose={onClose} />);
    unmount();
    pressEscape();
    expect(onClose).not.toHaveBeenCalled();
  });
});
