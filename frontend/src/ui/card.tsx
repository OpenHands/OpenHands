import { ReactNode } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "#/utils/utils";

const cardVariants = cva("flex", {
  variants: {
    variant: {
      default: "relative bg-[#26282D] border border-[#727987] rounded-xl",
      outlined: "relative bg-transparent border border-[#727987] rounded-xl",
      dark: "relative bg-black border border-[#242424] rounded-2xl",
    },
  },
  defaultVariants: {
    variant: "default",
  },
});

interface CardProps extends VariantProps<typeof cardVariants> {
  children?: ReactNode;
  className?: string;
  testId?: string;
}

export function Card({ children, className, testId, variant }: CardProps) {
  return (
    <div
      data-testid={testId}
      className={cn(cardVariants({ variant }), className)}
    >
      {children}
    </div>
  );
}
