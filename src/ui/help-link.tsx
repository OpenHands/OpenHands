import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "#/utils/utils";

const helpLinkVariants = cva("", {
  variants: {
    size: {
      default: "text-xs",
      settings: "text-sm text-[var(--oh-muted)] font-normal leading-5.5",
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
  /** Optional second link rendered after the suffix. */
  suffixLinkText?: string;
  suffixLinkHref?: string;
  /** Trailing text rendered after the second link (e.g. a period). */
  trailing?: string;
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
  suffixLinkText,
  suffixLinkHref,
  trailing,
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
      {suffixLinkText && suffixLinkHref ? (
        <>
          {" "}
          <a
            href={suffixLinkHref}
            rel="noreferrer noopener"
            target="_blank"
            className={cn(
              "underline underline-offset-2",
              helpLinkVariants({ size, linkColor }),
              linkTextClassName,
            )}
          >
            {suffixLinkText}
          </a>
          {trailing}
        </>
      ) : null}
    </p>
  );
}
