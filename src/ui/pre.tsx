import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "#/utils/utils";

const preVariants = cva("whitespace-pre-wrap", {
  variants: {
    size: {
      default: "text-sm",
      small: "text-xs",
    },
    font: {
      default: "",
      mono: "font-mono",
    },
    lineHeight: {
      default: "",
      relaxed: "leading-relaxed",
    },
    background: {
      default: "",
      dark: "bg-base",
    },
    textColor: {
      default: "",
      light: "text-[var(--oh-text-tertiary)]",
    },
    padding: {
      default: "",
      medium: "p-3",
      large: "px-5",
    },
    borderRadius: {
      default: "",
      medium: "rounded-md",
    },
    shadow: {
      default: "",
      inner: "shadow-inner",
    },
    maxHeight: {
      default: "",
      small: "max-h-[400px]",
      large: "max-h-[60vh]",
    },
    overflow: {
      default: "",
      auto: "overflow-auto",
    },
  },
  defaultVariants: {
    size: "default",
    font: "default",
    lineHeight: "default",
    background: "default",
    textColor: "default",
    padding: "default",
    borderRadius: "default",
    shadow: "default",
    maxHeight: "default",
    overflow: "default",
  },
});

interface PreProps extends VariantProps<typeof preVariants> {
  className?: string;
  testId?: string;
  children: React.ReactNode;
}

export function Pre({
  size,
  font,
  lineHeight,
  background,
  textColor,
  padding,
  borderRadius,
  shadow,
  maxHeight,
  overflow,
  className,
  testId,
  children,
}: PreProps) {
  return (
    <pre
      data-testid={testId}
      className={cn(
        preVariants({
          size,
          font,
          lineHeight,
          background,
          textColor,
          padding,
          borderRadius,
          shadow,
          maxHeight,
          overflow,
        }),
        className,
      )}
    >
      {children}
    </pre>
  );
}
