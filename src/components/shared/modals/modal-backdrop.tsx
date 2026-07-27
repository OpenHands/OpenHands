import React from "react";
import { createPortal } from "react-dom";

interface ModalBackdropProps {
  children: React.ReactNode;
  onClose?: () => void;
  /** When false, pressing Escape does not close the modal. Defaults to true. */
  closeOnEscape?: boolean;
  /** When false, clicking the backdrop does not close the modal. Defaults to true. */
  closeOnBackdropClick?: boolean;
  /** When true, renders above the default modal layer so it stacks over other
   *  modals (used by the telemetry consent banner over the onboarding modal).
   *  Defaults to false. */
  elevated?: boolean;
  "aria-label"?: string;
}

export function ModalBackdrop({
  children,
  onClose,
  closeOnEscape = true,
  closeOnBackdropClick = true,
  elevated = false,
  "aria-label": ariaLabel,
}: ModalBackdropProps) {
  React.useEffect(() => {
    if (!closeOnEscape) return undefined;
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose?.();
    };

    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [closeOnEscape, onClose]);

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!closeOnBackdropClick) return;
    if (e.target === e.currentTarget) onClose?.(); // only close if the click was on the backdrop
  };

  if (typeof document === "undefined") return null;

  // Portal to document.body so the modal's `position: fixed` resolves
  // against the viewport. Otherwise a transformed ancestor (e.g. the
  // onboarding slide rail) would become the containing block and the
  // modal would render trapped inside it instead of overlapping its
  // parent modal.
  return createPortal(
    <div
      role="dialog"
      aria-modal="true"
      aria-label={ariaLabel}
      className={`fixed inset-0 flex items-center justify-center ${
        elevated ? "z-[70]" : "z-60"
      }`}
    >
      <div
        onClick={handleClick}
        className="fixed inset-0 bg-black opacity-60"
      />
      <div className="relative">{children}</div>
    </div>,
    document.body,
  );
}
