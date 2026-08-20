import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";

describe("ModalBackdrop", () => {
  it("portals out of a transformed ancestor so position: fixed resolves against the viewport", () => {
    // Arrange: a transformed ancestor would otherwise become the
    // containing block for `position: fixed` descendants and trap the
    // modal inside it (the OnboardingModal / InstallServerModal bug).
    render(
      <div data-testid="transformed-ancestor" style={{ transform: "translateX(0)" }}>
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
