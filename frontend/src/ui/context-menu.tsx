import React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "#/utils/utils";

const contextMenuVariants = cva("text-white overflow-hidden z-50", {
  variants: {
    theme: {
      default:
        "absolute bg-tertiary rounded-[6px] context-menu-box-shadow py-[6px] px-1",
      naked: "relative",
    },
    size: {
      compact: "py-1 px-1",
      default: "",
    },
    layout: {
      vertical: "flex flex-col gap-2",
    },
    position: {
      top: "bottom-full",
      bottom: "top-full",
    },
    spacing: {
      default: "mt-2",
      none: "",
    },
    alignment: {
      left: "left-0",
      right: "right-0",
    },
  },
  compoundVariants: [
    {
      theme: "naked",
      className: "shadow-none",
    },
  ],
  defaultVariants: {
    theme: "default",
    size: "default",
    layout: "vertical",
    spacing: "default",
  },
});

interface ContextMenuProps {
  ref?: React.RefObject<HTMLUListElement | null>;
  testId?: string;
  children: React.ReactNode;
  className?: React.HTMLAttributes<HTMLUListElement>["className"];
  theme?: VariantProps<typeof contextMenuVariants>["theme"];
  size?: VariantProps<typeof contextMenuVariants>["size"];
  layout?: VariantProps<typeof contextMenuVariants>["layout"];
  position?: VariantProps<typeof contextMenuVariants>["position"];
  spacing?: VariantProps<typeof contextMenuVariants>["spacing"];
  alignment?: VariantProps<typeof contextMenuVariants>["alignment"];
}

export function ContextMenu({
  testId,
  children,
  className,
  ref,
  theme,
  size,
  layout,
  position,
  spacing,
  alignment,
}: ContextMenuProps) {
  return (
    <ul
      data-testid={testId}
      ref={ref}
      className={cn(
        contextMenuVariants({
          theme,
          size,
          layout,
          position,
          spacing,
          alignment,
        }),
        className,
      )}
    >
      {children}
    </ul>
  );
}
