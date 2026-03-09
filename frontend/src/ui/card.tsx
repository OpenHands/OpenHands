import { ReactNode } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "#/utils/utils";

const cardVariants = cva("flex", {
  variants: {
    w: {
      default: "w-auto",
      full: "w-full",
    },
    h: {
      default: "h-auto",
      full: "h-full",
      sm: "h-[263.5px]",
    },
    color: {
      default: "bg-[#26282D]",
    },
    gap: {
      default: "gap-[10px]",
      lg: "gap-6",
    },
    border: {
      none: "border-0",
      default: "border border-[#727987] rounded-[12px] ",
    },
  },
  defaultVariants: {
    w: "full",
    h: "default",
    color: "default",
    gap: "default",
    border: "default",
  },
});

interface CardProps extends VariantProps<typeof cardVariants> {
  children?: ReactNode;
  className?: string;
  testId?: string;
}

export function Card({
  children,
  className,
  testId,
  w,
  h,
  color,
  gap,
  border,
}: CardProps) {
  return (
    <div
      data-testid={testId}
      className={cn(cardVariants({ w, h, color, gap, border }), className)}
    >
      {children}
    </div>
  );
}
