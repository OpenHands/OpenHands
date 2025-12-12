import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "#/utils/utils";

const helpLinkVariants = cva("", {
  variants: {
    size: {
      default: "text-xs",
      settings: "text-sm text-[#A3A3A3] font-normal leading-5.5",
    },
    linkColor: {
      default: "",
      white: "text-white",
    },
  },
  defaultVariants: {
    size: "default",
    linkColor: "default",
  },
});

interface HelpLinkProps extends VariantProps<typeof helpLinkVariants> {
  testId: string;
  text: string;
  linkText: string;
  href: string;
  suffix?: string;
  className?: string;
  linkTextClassName?: string;
  suffixClassName?: string;
}

export function HelpLink({
  testId,
  text,
  linkText,
  href,
  suffix,
  size,
  linkColor,
  className,
  linkTextClassName,
  suffixClassName,
}: HelpLinkProps) {
  return (
    <p
      data-testid={testId}
      className={cn(helpLinkVariants({ size }), className)}
    >
      {text}{" "}
      <a
        href={href}
        rel="noreferrer noopener"
        target="_blank"
        className={cn(
          "underline underline-offset-2",
          helpLinkVariants({ size, linkColor }),
          linkTextClassName,
        )}
      >
        {linkText}
      </a>
      {suffix && <span className={suffixClassName}>{suffix}</span>}
    </p>
  );
}
