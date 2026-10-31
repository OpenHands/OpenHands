import React from "react";
import { cn } from "#/utils/utils";
import { clampLeftToViewport } from "#/hooks/use-popover-fixed-placement";

interface ToolsSubmenuFlyoutProps {
  /** Tailwind group name of the trigger row, e.g. "macros" for `group/macros`. */
  group: "agent-profile" | "git" | "macros";
  /** Vertical offset used while the flyout fits beside the trigger row. */
  sideTopClassName: string;
  isOpen: boolean;
  children: React.ReactNode;
}

const GROUP_HOVER_CLASSNAMES: Record<ToolsSubmenuFlyoutProps["group"], string> =
  {
    "agent-profile":
      "group-hover/agent-profile:opacity-100 group-hover/agent-profile:visible group-hover/agent-profile:pointer-events-auto",
    git: "group-hover/git:opacity-100 group-hover/git:visible group-hover/git:pointer-events-auto",
    macros:
      "group-hover/macros:opacity-100 group-hover/macros:visible group-hover/macros:pointer-events-auto",
  };

/**
 * A tools-menu submenu that opens to the right of its trigger row. When the
 * viewport has no room on the right (phone widths), it drops below the row
 * and slides left just far enough to stay on screen, overlapping the parent
 * menu instead of running off the edge.
 */
export function ToolsSubmenuFlyout({
  group,
  sideTopClassName,
  isOpen,
  children,
}: ToolsSubmenuFlyoutProps) {
  const ref = React.useRef<HTMLDivElement>(null);
  const [shift, setShift] = React.useState(0);

  const measure = React.useCallback(() => {
    const wrapper = ref.current;
    if (!wrapper) return;
    // The submenu is mounted (invisible) whenever the parent menu is open, so
    // it can be measured before it is revealed by hover or click.
    const content =
      wrapper.firstElementChild instanceof HTMLElement
        ? wrapper.firstElementChild
        : wrapper;
    const row = wrapper.parentElement;
    if (!row) return;
    // Offsets ignore the shift's (animated) transform, so this is the
    // beside-the-row position regardless of what is currently rendered.
    const naturalLeft =
      row.getBoundingClientRect().left +
      wrapper.offsetLeft +
      (content === wrapper ? 0 : content.offsetLeft);
    const width = content.offsetWidth;
    const next = clampLeftToViewport(naturalLeft, width) - naturalLeft;
    setShift(Math.min(0, next));
  }, []);

  React.useLayoutEffect(() => {
    measure();
    window.addEventListener("resize", measure);
    // Submenu content can change width after mount (e.g. a list that loads).
    const content = ref.current?.firstElementChild;
    const observer =
      content && typeof ResizeObserver !== "undefined"
        ? new ResizeObserver(measure)
        : null;
    if (content) observer?.observe(content);
    return () => {
      window.removeEventListener("resize", measure);
      observer?.disconnect();
    };
  }, [measure]);

  React.useLayoutEffect(() => {
    if (isOpen) measure();
  }, [isOpen, measure]);

  const isOverlay = shift < 0;

  return (
    <div
      ref={ref}
      data-testid={`${group}-submenu-flyout`}
      className={cn(
        "absolute left-full z-60 opacity-0 invisible pointer-events-none transition-all duration-200 ml-px",
        isOverlay
          ? "top-full -mt-2 translate-x-(--tools-submenu-shift)"
          : sideTopClassName,
        GROUP_HOVER_CLASSNAMES[group],
        "hover:opacity-100 hover:visible hover:pointer-events-auto",
        isOpen && "opacity-100 visible pointer-events-auto",
      )}
      style={
        isOverlay
          ? ({ "--tools-submenu-shift": `${shift}px` } as React.CSSProperties)
          : undefined
      }
      onFocusCapture={measure}
    >
      {children}
    </div>
  );
}
