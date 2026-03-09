import React from "react";
import { cn } from "#/utils/utils";

interface ElementProps {
  children?: React.ReactNode;
  className?: string;
}

interface ListProps extends ElementProps {
  start?: number;
}

interface AnchorProps extends ElementProps {
  href?: string;
}

/**
 * Creates custom markdown components for plan views with reduced font sizes and tighter spacing.
 * Accepts an optional extraClassName that will be applied to all elements.
 */
export function createPlanComponents(extraClassName?: string) {
  return {
    h1: ({ children, className }: ElementProps) => (
      <h1
        className={cn(
          "text-lg text-white font-bold leading-6 mb-1.5 mt-3 first:mt-0",
          className,
          extraClassName,
        )}
      >
        {children}
      </h1>
    ),
    h2: ({ children, className }: ElementProps) => (
      <h2
        className={cn(
          "text-base font-semibold leading-5 text-white mb-1 mt-2.5 first:mt-0",
          className,
          extraClassName,
        )}
      >
        {children}
      </h2>
    ),
    h3: ({ children, className }: ElementProps) => (
      <h3
        className={cn(
          "text-sm font-semibold text-white mb-1 mt-2 first:mt-0",
          className,
          extraClassName,
        )}
      >
        {children}
      </h3>
    ),
    h4: ({ children, className }: ElementProps) => (
      <h4
        className={cn(
          "text-sm font-semibold text-white mb-1 mt-2 first:mt-0",
          className,
          extraClassName,
        )}
      >
        {children}
      </h4>
    ),
    h5: ({ children, className }: ElementProps) => (
      <h5
        className={cn(
          "text-xs font-semibold text-white mb-0.5 mt-1.5 first:mt-0",
          className,
          extraClassName,
        )}
      >
        {children}
      </h5>
    ),
    h6: ({ children, className }: ElementProps) => (
      <h6
        className={cn(
          "text-xs font-medium text-gray-300 mb-0.5 mt-1.5 first:mt-0",
          className,
          extraClassName,
        )}
      >
        {children}
      </h6>
    ),
    p: ({ children, className }: ElementProps) => (
      <p
        className={cn("py-2.5 first:pt-0 last:pb-0", className, extraClassName)}
      >
        {children}
      </p>
    ),
    ul: ({ children, className }: ElementProps) => (
      <ul
        className={cn(
          "list-disc ml-5 pl-2 whitespace-normal",
          className,
          extraClassName,
        )}
      >
        {children}
      </ul>
    ),
    ol: ({ children, className, start }: ListProps) => (
      <ol
        className={cn(
          "list-decimal ml-5 pl-2 whitespace-normal",
          className,
          extraClassName,
        )}
        start={start}
      >
        {children}
      </ol>
    ),
    li: ({ children, className }: ElementProps) => (
      <li className={cn(className, extraClassName)}>{children}</li>
    ),
    a: ({ children, className, href }: AnchorProps) => (
      <a
        className={cn(
          "text-blue-500 hover:underline",
          className,
          extraClassName,
        )}
        href={href}
        target="_blank"
        rel="noopener noreferrer"
      >
        {children}
      </a>
    ),
    code: ({ children, className }: ElementProps) => (
      <code
        className={cn(
          "bg-[#2a3038] px-1.5 py-0.5 rounded text-[#e6edf3] border border-[#30363d]",
          className,
          extraClassName,
        )}
      >
        {children}
      </code>
    ),
  };
}

// Default plan components without extra className (for backward compatibility)
export const planComponents = createPlanComponents();

/**
 * @deprecated Use planComponents instead
 */
export const planHeadings = planComponents;
