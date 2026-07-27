import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "#/utils/utils";

/** 12px slot (4px + 1px line + 4px at default rem) inside `px-1` menus/popovers. */
export const MENU_DIVIDER_VERTICAL_CLASS = "h-3";

const dividerVariants = cva(
  "shrink-0 self-stretch min-w-full bg-[var(--oh-border)]",
  {
    variants: {
      orientation: {
        horizontal: "h-[1px]",
      },
      color: {
        light: "bg-[var(--oh-border)]",
      },
      size: {
        thin: "h-[1px]",
      },
      inset: {
        none: "",
        menu: "",
      },
    },
    defaultVariants: {
      orientation: "horizontal",
      color: "light",
      size: "thin",
      inset: "none",
    },
  },
);

interface DividerProps extends VariantProps<typeof dividerVariants> {
  className?: string;
  testId?: string;
}

export function Divider({
  orientation,
  color,
  size,
  inset,
  className,
  testId,
}: DividerProps) {
  if (inset === "menu") {
    return (
      <div
        data-testid={testId}
        role="separator"
        className={cn(
          "relative min-w-full shrink-0 self-stretch",
          MENU_DIVIDER_VERTICAL_CLASS,
          className,
        )}
      >
        <div
          aria-hidden
          className="absolute top-1/2 -left-1 -right-1 h-px -translate-y-1/2 bg-[var(--oh-border)]"
        />
      </div>
    );
  }

  return (
    <div
      data-testid={testId}
      role="separator"
      className={cn(
        dividerVariants({ orientation, color, size, inset }),
        className,
      )}
    />
  );
}
