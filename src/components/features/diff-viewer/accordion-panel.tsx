import type { ReactNode } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { cn } from "#/utils/utils";

const ACCORDION_TRANSITION = {
  duration: 0.2,
  ease: "easeInOut" as const,
};

interface AccordionPanelProps {
  open: boolean;
  children: ReactNode;
  testId?: string;
  className?: string;
}

/**
 * Height/opacity expand-collapse wrapper for Diffs / Commits accordion rows.
 * Honors prefers-reduced-motion by skipping animation.
 */
export function AccordionPanel({
  open,
  children,
  testId,
  className,
}: AccordionPanelProps) {
  const reduceMotion = useReducedMotion();

  if (reduceMotion) {
    return open ? (
      <div data-testid={testId} className={className}>
        {children}
      </div>
    ) : null;
  }

  return (
    <AnimatePresence initial={false}>
      {open ? (
        <motion.div
          key="accordion-panel"
          data-testid={testId}
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: "auto", opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          transition={ACCORDION_TRANSITION}
          className={cn("overflow-hidden", className)}
        >
          {children}
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
