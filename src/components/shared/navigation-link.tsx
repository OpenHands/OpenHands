import React from "react";
import {
  useNavigation,
  type NavigationOptions,
} from "#/context/navigation-context";

interface NavigationLinkClassNameState {
  isActive: boolean;
}

export interface NavigationLinkProps extends Omit<
  React.AnchorHTMLAttributes<HTMLAnchorElement>,
  "className" | "href"
> {
  to: string;
  replace?: boolean;
  end?: boolean;
  className?:
    | string
    | ((state: NavigationLinkClassNameState) => string | undefined);
}

function isModifiedEvent(event: React.MouseEvent<HTMLAnchorElement>) {
  return event.metaKey || event.altKey || event.ctrlKey || event.shiftKey;
}

function isPathActive(currentPath: string, to: string, end: boolean) {
  if (to === "/") {
    return currentPath === to;
  }

  if (end) {
    return currentPath === to;
  }

  return currentPath === to || currentPath.startsWith(`${to}/`);
}

export const NavigationLink = React.forwardRef<
  HTMLAnchorElement,
  NavigationLinkProps
>(
  (
    {
      to,
      replace = false,
      end = false,
      onClick,
      className,
      children,
      target,
      rel,
      ...props
    },
    ref,
  ) => {
    const { currentPath, navigate } = useNavigation();
    const isActive = isPathActive(currentPath, to, end);

    const resolvedClassName =
      typeof className === "function" ? className({ isActive }) : className;

    const handleClick = (event: React.MouseEvent<HTMLAnchorElement>) => {
      onClick?.(event);

      if (
        event.defaultPrevented ||
        event.button !== 0 ||
        isModifiedEvent(event) ||
        target === "_blank"
      ) {
        return;
      }

      event.preventDefault();
      navigate(to, { replace } satisfies NavigationOptions);
    };

    return (
      <a
        {...props}
        ref={ref}
        href={to}
        target={target}
        rel={rel}
        onClick={handleClick}
        className={resolvedClassName}
        aria-current={isActive ? "page" : undefined}
      >
        {children}
      </a>
    );
  },
);

NavigationLink.displayName = "NavigationLink";
