import { Tooltip as HeroTooltip } from "@heroui/react";
import React, { type ReactElement, type ReactNode } from "react";
import { mergeProps, mergeRefs } from "react-aria";
import { cn } from "#/utils/utils";

export interface TooltipProps {
  children: ReactNode;
  content: ReactNode;
  placement?:
    | React.ComponentProps<typeof HeroTooltip.Content>["placement"]
    | "top-start"
    | "top-end"
    | "bottom-start"
    | "bottom-end"
    | "left-start"
    | "left-end"
    | "right-start"
    | "right-end";
  offset?: number;
  shouldFlip?: boolean;
  showArrow?: boolean;
  delay?: number;
  closeDelay?: number;
  isOpen?: boolean;
  isDisabled?: boolean;
  disableAnimation?: boolean;
  className?: string;
  classNames?: { content?: string };
}

type AriaPlacement = React.ComponentProps<
  typeof HeroTooltip.Content
>["placement"];

// React Aria aligns a left or right tooltip along the vertical axis, so its
// placements are "right top" and "right bottom"; "right start" is not one and
// falls back to centering on the trigger.
const SIDE_ALIGNMENT: Record<string, string> = {
  "left-start": "left top",
  "left-end": "left bottom",
  "right-start": "right top",
  "right-end": "right bottom",
};

export function toAriaPlacement(
  placement: TooltipProps["placement"],
): AriaPlacement {
  if (!placement) return undefined;
  return (SIDE_ALIGNMENT[placement] ??
    placement.replace("-", " ")) as AriaPlacement;
}

export function Tooltip({
  children,
  content,
  placement,
  offset,
  shouldFlip,
  showArrow,
  delay = 0,
  closeDelay,
  isOpen,
  isDisabled,
  disableAnimation,
  className,
  classNames,
}: TooltipProps) {
  const child = (
    React.isValidElement(children) ? children : <span>{children}</span>
  ) as ReactElement<
    React.HTMLAttributes<HTMLElement> & { ref?: React.Ref<HTMLElement> }
  >;

  return (
    <HeroTooltip
      delay={delay}
      closeDelay={closeDelay}
      isOpen={isOpen}
      isDisabled={isDisabled}
      shouldSkipAnimation={disableAnimation}
    >
      <HeroTooltip.Trigger
        render={({
          ref,
          role: _role,
          tabIndex: _tabIndex,
          className: _className,
          ...triggerProps
        }) =>
          // Keep the child's semantics, tab order and layout; only compose tooltip events.
          React.cloneElement(child, {
            ...mergeProps(triggerProps, child.props),
            ref: mergeRefs(ref, child.props.ref),
          })
        }
      />
      <HeroTooltip.Content
        placement={toAriaPlacement(placement)}
        offset={offset}
        shouldFlip={shouldFlip}
        showArrow={showArrow}
        className={cn(className, classNames?.content)}
      >
        {showArrow && <HeroTooltip.Arrow />}
        {content}
      </HeroTooltip.Content>
    </HeroTooltip>
  );
}
