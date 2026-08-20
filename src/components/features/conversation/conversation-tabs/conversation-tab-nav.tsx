import { ComponentType } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { cn } from "#/utils/utils";

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
}: ConversationTabNavProps) {
  const reduceMotion = useReducedMotion();
  const disableAnimation =
    measureOnly || reduceMotion || import.meta.env.MODE === "test";
  const enableLayoutAnimation = !disableAnimation && !suppressLayoutAnimation;

  const buttonClassName = cn(
    "flex items-center rounded-md cursor-pointer",
    "pl-1.5 pr-2 py-1 lg:py-1.5",
    "text-[var(--oh-muted)] bg-transparent",
    isActive && "bg-[var(--oh-interactive-active)] text-white",
    isActive
      ? "hover:text-white hover:bg-[var(--oh-interactive-hover)]"
      : "hover:text-white hover:bg-white/5",
    isActive
      ? "focus-within:text-white"
      : "focus-within:text-[var(--oh-muted)]",
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
        {...(measureOnly
          ? {}
          : { "data-testid": `conversation-tab-${tabValue}` as const })}
        data-tab-measure={measureOnly ? "true" : undefined}
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
      {...(measureOnly
        ? {}
        : { "data-testid": `conversation-tab-${tabValue}` as const })}
      data-tab-measure={measureOnly ? "true" : undefined}
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
