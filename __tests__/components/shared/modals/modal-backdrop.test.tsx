import React from "react";
import { createPortal } from "react-dom";
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";

describe("ModalBackdrop", () => {
  it("locks body scrolling while open and restores the previous overflow after unmount", () => {
    document.body.style.overflow = "auto";

    const { unmount } = render(
      <ModalBackdrop>
        <div />
      </ModalBackdrop>,
    );

    expect(document.body.style.overflow).toBe("hidden");

    unmount();

    expect(document.body.style.overflow).toBe("auto");
  });

  it("portals out of a transformed ancestor so position: fixed resolves against the viewport", () => {
    // Arrange: a transformed ancestor would otherwise become the
    // containing block for `position: fixed` descendants and trap the
    // modal inside it (the OnboardingModal / InstallServerModal bug).
    render(
      <div
        data-testid="transformed-ancestor"
        style={{ transform: "translateX(0)" }}
      >
        <ModalBackdrop onClose={vi.fn()}>
          <p>modal content</p>
        </ModalBackdrop>
      </div>,
    );

    // Act
    const dialog = screen.getByRole("dialog");
    const transformedAncestor = screen.getByTestId("transformed-ancestor");

    // Assert
    expect(transformedAncestor.contains(dialog)).toBe(false);
    expect(document.body.contains(dialog)).toBe(true);
  });

  it("calls onClose when the user presses Escape", async () => {
    // Arrange
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <ModalBackdrop onClose={onClose}>
        <p>modal content</p>
      </ModalBackdrop>,
    );

    // Act
    await user.keyboard("{Escape}");

    // Assert
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("calls onClose when the backdrop is clicked but not when the content is clicked", async () => {
    // Arrange
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <ModalBackdrop onClose={onClose}>
        <button type="button">inside content</button>
      </ModalBackdrop>,
    );

    // Act: click the content first — this must NOT close.
    await user.click(screen.getByRole("button", { name: "inside content" }));
    const callsAfterContentClick = onClose.mock.calls.length;

    // Act: now click the backdrop overlay (the dialog root's backdrop child).
    const dialog = screen.getByRole("dialog");
    const backdrop = dialog.firstElementChild as HTMLElement;
    await user.click(backdrop);

    // Assert
    expect(callsAfterContentClick).toBe(0);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("does not call onClose when the backdrop is clicked and closeOnBackdropClick is false", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <ModalBackdrop onClose={onClose} closeOnBackdropClick={false}>
        <button type="button">inside content</button>
      </ModalBackdrop>,
    );

    const dialog = screen.getByRole("dialog");
    const backdrop = dialog.firstElementChild as HTMLElement;
    await user.click(backdrop);

    expect(onClose).not.toHaveBeenCalled();
  });

  it("stacks above the default modal layer when elevated", () => {
    // Arrange: a default backdrop (the layer the onboarding modal uses) and an
    // elevated backdrop (the layer the telemetry consent banner opts into).
    render(
      <>
        <ModalBackdrop aria-label="default-modal">
          <p>default modal</p>
        </ModalBackdrop>
        <ModalBackdrop elevated aria-label="elevated-modal">
          <p>elevated modal</p>
        </ModalBackdrop>
      </>,
    );

    // Act: read the numeric stacking layer each modal renders on. jsdom cannot
    // compute paint order, so compare the z-index the overlay carries.
    const layerOf = (name: string) =>
      Number(
        screen
          .getByRole("dialog", { name })
          .className.match(/z-\[?(\d+)\]?/)?.[1] ?? 0,
      );

    // Assert: the elevated modal paints above the default one regardless of
    // mount order, so the consent banner is never covered by the onboarding modal.
    expect(layerOf("elevated-modal")).toBeGreaterThan(layerOf("default-modal"));
  });
});

describe("ModalBackdrop focus management", () => {
  function PageWithModal({
    renderModalContent,
  }: {
    renderModalContent: (close: () => void) => React.ReactNode;
  }) {
    const [isOpen, setIsOpen] = React.useState(false);
    const close = () => setIsOpen(false);
    return (
      <>
        <button type="button" onClick={() => setIsOpen(true)}>
          open
        </button>
        <input aria-label="page composer" />
        {isOpen ? (
          <ModalBackdrop onClose={close} aria-label="test dialog">
            {renderModalContent(close)}
          </ModalBackdrop>
        ) : null}
      </>
    );
  }

  const threeButtons = (close: () => void) => (
    <>
      <button type="button" onClick={close}>
        first
      </button>
      <button type="button">middle</button>
      <button type="button">last</button>
    </>
  );

  it("moves focus to the first tabbable control inside the dialog when it opens", async () => {
    const user = userEvent.setup();
    render(<PageWithModal renderModalContent={threeButtons} />);

    await user.click(screen.getByRole("button", { name: "open" }));

    expect(screen.getByRole("button", { name: "first" })).toHaveFocus();
  });

  it("keeps focus on a control the dialog content focuses itself", async () => {
    const user = userEvent.setup();
    render(
      <PageWithModal
        renderModalContent={() => (
          <>
            <button type="button">first</button>
            {/* eslint-disable-next-line jsx-a11y/no-autofocus */}
            <input aria-label="name" autoFocus />
          </>
        )}
      />,
    );

    await user.click(screen.getByRole("button", { name: "open" }));

    expect(screen.getByRole("textbox", { name: "name" })).toHaveFocus();
  });

  it("keeps the content's own focus and restores the opener under StrictMode", async () => {
    const user = userEvent.setup();
    render(
      <React.StrictMode>
        <PageWithModal
          renderModalContent={(close) => (
            <>
              <button type="button" onClick={close}>
                first
              </button>
              {/* eslint-disable-next-line jsx-a11y/no-autofocus */}
              <input aria-label="name" autoFocus />
            </>
          )}
        />
      </React.StrictMode>,
    );
    const opener = screen.getByRole("button", { name: "open" });

    await user.click(opener);
    expect(screen.getByRole("textbox", { name: "name" })).toHaveFocus();

    await user.click(screen.getByRole("button", { name: "first" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(opener).toHaveFocus();
  });

  it("wraps Tab and Shift+Tab inside the dialog and skips inert controls", async () => {
    const user = userEvent.setup();
    render(
      <PageWithModal
        renderModalContent={(close) => (
          <>
            {threeButtons(close)}
            <div inert>
              <button type="button">hidden panel</button>
            </div>
          </>
        )}
      />,
    );
    await user.click(screen.getByRole("button", { name: "open" }));
    const first = screen.getByRole("button", { name: "first" });
    const last = screen.getByRole("button", { name: "last" });

    await user.tab({ shift: true });
    expect(last).toHaveFocus();

    await user.tab();
    expect(first).toHaveFocus();
  });

  it("pulls focus back when something behind the dialog takes it", async () => {
    const user = userEvent.setup();
    render(<PageWithModal renderModalContent={threeButtons} />);
    await user.click(screen.getByRole("button", { name: "open" }));

    screen.getByRole("textbox", { name: "page composer" }).focus();

    expect(screen.getByRole("button", { name: "first" })).toHaveFocus();
  });

  it("lets a popover the dialog content portals elsewhere keep focus", async () => {
    const user = userEvent.setup();
    render(
      <PageWithModal
        renderModalContent={() =>
          createPortal(<input aria-label="popover search" />, document.body)
        }
      />,
    );
    await user.click(screen.getByRole("button", { name: "open" }));
    const popoverInput = screen.getByRole("textbox", {
      name: "popover search",
    });

    popoverInput.focus();

    expect(popoverInput).toHaveFocus();
  });

  it("restores focus to the opener when the dialog closes", async () => {
    const user = userEvent.setup();
    render(<PageWithModal renderModalContent={threeButtons} />);
    const opener = screen.getByRole("button", { name: "open" });
    await user.click(opener);

    await user.click(screen.getByRole("button", { name: "first" }));

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(opener).toHaveFocus();
  });

  function OuterWithRemoval({ removesRow }: { removesRow: boolean }) {
    const [removed, setRemoved] = React.useState(false);
    const [confirming, setConfirming] = React.useState(false);
    return (
      <>
        <button type="button">outer first</button>
        {removed ? null : (
          <button type="button" onClick={() => setConfirming(true)}>
            remove
          </button>
        )}
        {confirming ? (
          <ConfirmationModal
            text="Remove it?"
            onConfirm={() => {
              setRemoved(removesRow);
              setConfirming(false);
            }}
            onCancel={() => setConfirming(false)}
          />
        ) : null}
      </>
    );
  }

  it("gives focus to the topmost of nested dialogs and hands it back on close", async () => {
    const user = userEvent.setup();
    render(
      <PageWithModal
        renderModalContent={() => <OuterWithRemoval removesRow={false} />}
      />,
    );
    await user.click(screen.getByRole("button", { name: "open" }));
    const removeButton = screen.getByRole("button", { name: "remove" });

    await user.click(removeButton);
    const cancel = screen.getByTestId("cancel-button");
    const confirm = screen.getByTestId("confirm-button");
    expect(cancel).toHaveFocus();
    await user.tab();
    expect(confirm).toHaveFocus();
    await user.tab();
    expect(cancel).toHaveFocus();
    await user.tab({ shift: true });
    expect(confirm).toHaveFocus();

    await user.click(cancel);
    expect(removeButton).toHaveFocus();
  });

  it("moves focus into the dialog below when the nested dialog's opener is gone", async () => {
    const user = userEvent.setup();
    render(
      <PageWithModal
        renderModalContent={() => <OuterWithRemoval removesRow />}
      />,
    );
    await user.click(screen.getByRole("button", { name: "open" }));
    await user.click(screen.getByRole("button", { name: "remove" }));

    await user.click(screen.getByTestId("confirm-button"));

    expect(screen.getByRole("button", { name: "outer first" })).toHaveFocus();
  });
});
