import React from "react";
import { cn } from "#/utils/utils";

/** Shared modal content widths (sm / md / lg / xl). */
export type ModalWidth = "sm" | "md" | "lg" | "xl";

export const MODAL_WIDTH_CLASS: Record<ModalWidth, string> = {
  sm: "w-[384px]",
  md: "w-[520px]",
  lg: "w-[640px]",
  xl: "w-[720px]",
};

export const MODAL_MAX_WIDTH_VIEWPORT = "max-w-[90vw]";

export function modalWidthClassName(width: ModalWidth): string {
  return MODAL_WIDTH_CLASS[width];
}

interface ModalBodyProps {
  testID?: string;
  children: React.ReactNode;
  className?: React.HTMLProps<HTMLDivElement>["className"];
  width?: ModalWidth;
}

export function ModalBody({
  testID,
  children,
  className,
  width = "sm",
}: ModalBodyProps) {
  return (
    <div
      data-testid={testID}
      className={cn(
        "bg-base-secondary flex flex-col gap-6 items-center p-6 rounded-xl",
        modalWidthClassName(width),
        className,
      )}
    >
      {children}
    </div>
  );
}
