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
    hover: {
      none: "",
      elevated: [
        "transition-all duration-200",
        "hover:bg-[linear-gradient(180deg,#0F0F0F_0%,#0A0A0A_100%)]",
        "hover:border-t-[#242424CC]",
        "hover:shadow-[0px_4px_6px_-4px_#0000001A,0px_10px_15px_-3px_#0000001A]",
        "before:absolute before:inset-0 before:rounded-2xl before:opacity-0 before:transition-opacity before:duration-200",
        "before:bg-[radial-gradient(98.4%_116.11%_at_50%_0%,rgba(255,255,255,0.08)_0%,rgba(0,0,0,0)_70%)]",
        "hover:before:opacity-100",
        "before:pointer-events-none",
      ].join(" "),
    },
    gradient: {
      none: "",
      standard: [
        "bg-[#0A0A0A80] border-t-[#24242499]",
        "shadow-[0px_4px_6px_-4px_#0000001A,0px_10px_15px_-3px_#0000001A]",
        "before:absolute before:inset-0 before:rounded-2xl before:pointer-events-none",
        "before:bg-[radial-gradient(144.32%_106.6%_at_50%_0%,rgba(255,255,255,0.14)_0%,rgba(0,0,0,0)_55%)]",
      ].join(" "),
    },
  },
  defaultVariants: {
    theme: "default",
    hover: "none",
    gradient: "none",
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
  theme,
  hover,
  gradient,
}: CardProps) {
  return (
    <div
      data-testid={testId}
      className={cn(cardVariants({ theme, hover, gradient }), className)}
    >
      {children}
    </div>
  );
}
