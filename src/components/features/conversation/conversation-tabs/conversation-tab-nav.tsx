import { ComponentType, KeyboardEvent, Ref } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { cn } from "#/utils/utils";
import {
  CONVERSATION_TAB_PANEL_ID,
  conversationTabId,
} from "./conversation-tab-ids";

const TAB_LABEL_MAX_WIDTH_PX = 160;

const tabLabelTransition = {
  duration: 0.22,
  ease: [0.4, 0, 0.2, 1] as const,
};

type ConversationTabNavProps = {
  tabValue: string;
  icon: ComponentType<{ className: string }>;
  onClick(): void;
  isActive?: boolean;
  label?: string;
  className?: string;
  /** Omit test id (e.g. offscreen width measurement clones). */
  measureOnly?: boolean;
  /** Disable layout-driven shifts while the drawer width is being dragged. */
  suppressLayoutAnimation?: boolean;
  /** Roving tabindex: only the strip's current tab stop gets `0`. */
  tabIndex?: number;
  onKeyDown?(event: KeyboardEvent<HTMLButtonElement>): void;
  buttonRef?: Ref<HTMLButtonElement>;
};

export function ConversationTabNav({
  tabValue,
  icon: Icon,
  onClick,
  isActive,
  label,
  className,
  measureOnly,
  suppressLayoutAnimation = false,
  tabIndex,
  onKeyDown,
  buttonRef,
}: ConversationTabNavProps) {
  const reduceMotion = useReducedMotion();
  const disableAnimation =
    measureOnly || reduceMotion || import.meta.env.MODE === "test";
  const enableLayoutAnimation = !disableAnimation && !suppressLayoutAnimation;

  // Measurement clones live in an `aria-hidden` row, so they stay out of the
  // tablist and out of the tab order. The real tabs carry an explicit name
  // because an inactive tab hides its label and would otherwise be announced
  // as an unlabelled button.
  const tabProps = measureOnly
    ? ({ "data-tab-measure": "true", tabIndex: -1 } as const)
    : ({
        // The tab's DOM id and its test id are the same string — the panel's
        // `aria-labelledby` and the tests address a tab the same way.
        "data-testid": conversationTabId(tabValue),
        id: conversationTabId(tabValue),
        role: "tab",
        "aria-selected": Boolean(isActive),
        "aria-controls": CONVERSATION_TAB_PANEL_ID,
        "aria-label": label,
        tabIndex,
        onKeyDown,
        ref: buttonRef,
      } as const);

  const buttonClassName = cn(
    "flex items-center rounded-md cursor-pointer",
    "pl-1.5 pr-2 py-1 lg:py-1.5",
    "text-muted bg-transparent",
    isActive && "bg-interactive-active text-contrast",
    isActive
      ? "hover:text-contrast hover:bg-interactive-hover"
      : "hover:text-contrast hover:bg-contrast/5",
    isActive ? "focus-within:text-contrast" : "focus-within:text-muted",
    className,
  );

  const iconElement = <Icon className={cn("h-4 w-4 shrink-0 text-inherit")} />;

  const labelElement =
    label && isActive ? (
      <span className="whitespace-nowrap text-sm font-normal">{label}</span>
    ) : null;

  const animatedLabelElement = label ? (
    <motion.span
      initial={false}
      animate={{
        maxWidth: isActive ? TAB_LABEL_MAX_WIDTH_PX : 0,
        opacity: isActive ? 1 : 0,
        marginLeft: isActive ? 8 : 0,
      }}
      transition={tabLabelTransition}
      className="block overflow-hidden whitespace-nowrap text-sm font-normal"
      aria-hidden={!isActive}
    >
      {label}
    </motion.span>
  ) : null;

  if (disableAnimation) {
    return (
      <button
        type="button"
        onClick={onClick}
        {...tabProps}
        className={cn(buttonClassName, "gap-2")}
      >
        {iconElement}
        {labelElement}
      </button>
    );
  }

  return (
    <motion.button
      layout={enableLayoutAnimation ? "position" : false}
      type="button"
      onClick={onClick}
      {...tabProps}
      className={buttonClassName}
      transition={
        enableLayoutAnimation ? { layout: tabLabelTransition } : undefined
      }
    >
      {iconElement}
      {animatedLabelElement}
    </motion.button>
  );
}
