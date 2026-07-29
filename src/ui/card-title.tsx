import { ReactNode } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "#/utils/utils";

const cardTitleVariants = cva("flex items-center", {
  variants: {
    gap: {
      default: "gap-[10px]",
    },
    textSize: {
      default: "text-base",
    },
    fontWeight: {
      default: "font-medium",
    },
    textColor: {
      default: "text-white",
    },
    lineHeight: {
      default: "leading-5",
    },
  },
  defaultVariants: {
    gap: "default",
    textSize: "default",
    fontWeight: "default",
    textColor: "default",
    lineHeight: "default",
  },
});

interface CardTitleProps extends VariantProps<typeof cardTitleVariants> {
  icon?: ReactNode;
  children: ReactNode;
  className?: string;
}

export function CardTitle({
  icon,
  children,
  className = "",
  gap,
  textSize,
  fontWeight,
  textColor,
  lineHeight,
}: CardTitleProps) {
  return (
    <div
      className={cn(
        cardTitleVariants({
          gap,
          textSize,
          fontWeight,
          textColor,
          lineHeight,
        }),
        className,
      )}
    >
      {icon}
      <span
        className={cn(
          cardTitleVariants({
            lineHeight,
            textSize,
            fontWeight,
            textColor,
          }),
          "flex items-center",
        )}
      >
        {children}
      </span>
    </div>
  );
}
