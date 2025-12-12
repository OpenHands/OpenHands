import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "#/utils/utils";

const dividerVariants = cva("w-full", {
  variants: {
    orientation: {
      horizontal: "h-[1px]",
    },
    color: {
      light: "bg-[#5C5D62]",
    },
    size: {
      thin: "h-[1px]",
    },
  },
  defaultVariants: {
    orientation: "horizontal",
    color: "light",
    size: "thin",
  },
});

interface DividerProps extends VariantProps<typeof dividerVariants> {
  className?: string;
  testId?: string;
}

export function Divider({
  orientation,
  color,
  size,
  className,
  testId,
}: DividerProps) {
  return (
    <div
      data-testid={testId}
      className={cn(dividerVariants({ orientation, color, size }), className)}
    />
  );
}
