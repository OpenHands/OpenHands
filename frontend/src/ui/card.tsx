import { ReactNode } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "#/utils/utils";

const cardVariants = cva("flex", {
  variants: {
    theme: {
      default: "relative bg-[#26282D] border border-[#727987] rounded-xl",
      outlined: "relative bg-transparent border border-[#727987] rounded-xl",
      dark: "relative bg-black border border-[#242424] rounded-2xl",
    },
  },
  defaultVariants: {
    theme: "default",
  },
});

interface CardProps extends VariantProps<typeof cardVariants> {
  children?: ReactNode;
  className?: string;
  testId?: string;
}

export function Card({ children, className, testId, theme }: CardProps) {
  return (
    <div
      data-testid={testId}
      className={cn(cardVariants({ theme }), className)}
    >
      {children}
    </div>
  );
}
