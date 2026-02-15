import { ReactNode } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "#/utils/utils";

const cardVariants = cva(
  "w-full flex flex-col rounded-xl p-5 border border-[#27272A] bg-[#18181B] relative transition-all duration-200 hover:border-[#3F3F46] hover:shadow-[0_4px_24px_rgba(0,0,0,0.2)]",
  {
    variants: {
      gap: {
        default: "gap-[10px]",
        large: "gap-6",
      },
      minHeight: {
        default: "min-h-[286px] md:min-h-auto",
        small: "min-h-[263.5px]",
      },
    },
    defaultVariants: {
      gap: "default",
      minHeight: "default",
    },
  },
);

interface CardProps extends VariantProps<typeof cardVariants> {
  children: ReactNode;
  className?: string;
  testId?: string;
}

export function Card({
  children,
  className = "",
  testId,
  gap,
  minHeight,
}: CardProps) {
  return (
    <div
      data-testid={testId}
      className={cn(cardVariants({ gap, minHeight }), className)}
    >
      {children}
    </div>
  );
}
