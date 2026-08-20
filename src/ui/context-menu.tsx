import React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "#/utils/utils";
import { dropdownMenuListGapClassName } from "#/utils/dropdown-classes";

const contextMenuVariants = cva(
  "z-50 overflow-hidden text-[var(--oh-foreground)]",
  {
    variants: {
      theme: {
        default:
          "absolute rounded-md border border-[var(--oh-border-subtle)] bg-tertiary px-1 py-1 shadow-lg",
        naked: "relative",
        /** In document-body portal; coordinates come from inline `style`. */
        popover:
          "relative rounded-md border border-[var(--oh-border-subtle)] bg-tertiary px-1 py-1 shadow-lg",
      },
      size: {
        compact: "py-1 px-1",
        default: "",
      },
      layout: {
        vertical: cn("flex flex-col", dropdownMenuListGapClassName),
      },
      position: {
        top: "bottom-full",
        bottom: "top-full",
        none: "",
      },
      spacing: {
        default: "mt-2",
        none: "",
      },
      alignment: {
        left: "left-0",
        right: "right-0",
        none: "",
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
  },
);

interface ContextMenuProps {
  ref?: React.RefObject<HTMLUListElement | null>;
  testId?: string;
  children: React.ReactNode;
  className?: React.HTMLAttributes<HTMLUListElement>["className"];
  style?: React.CSSProperties;
  onKeyDown?: React.KeyboardEventHandler<HTMLUListElement>;
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
  style,
  onKeyDown,
  ref,
  theme,
  size,
  layout,
  position,
  spacing,
  alignment,
}: ContextMenuProps) {
  return (
    // Keyboard navigation is opt-in via `onKeyDown` (home automation kebab).
    // eslint-disable-next-line jsx-a11y/no-noninteractive-element-interactions -- popup list hosts Escape/arrow handlers when provided
    <ul
      data-testid={testId}
      data-position={position}
      ref={ref}
      style={style}
      onKeyDown={onKeyDown}
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
