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

/**
 * `currentPath` is a pathname, so compare against the pathname portion of
 * `to`. Links may carry a query string (e.g. the backend identity pinned on
 * sidebar conversation links) that must not defeat the active check.
 */
function toPathname(to: string) {
  return to.split(/[?#]/)[0];
}

function isPathActive(currentPath: string, to: string, end: boolean) {
  const path = toPathname(to);

  if (path === "/") {
    return currentPath === path;
  }

  if (end) {
    return currentPath === path;
  }

  return currentPath === path || currentPath.startsWith(`${path}/`);
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
