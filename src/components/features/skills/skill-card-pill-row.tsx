import React from "react";
import { createPortal } from "react-dom";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";

/** @deprecated Use {@link extensionModuleCardPillClassName} from `#/utils/extension-module-card-classes`. */
export const SKILL_CARD_PILL_CLASS = extensionModuleCardPillClassName;

const PILL_GAP_PX = 6;
const OVERFLOW_PILL_WIDTH_PX = 40;
const OVERFLOW_POPOVER_GUTTER_PX = 8;
const OVERFLOW_POPOVER_OFFSET_PX = 4;

function placePopoverByTrigger(
  trigger: DOMRect,
  popoverWidth: number,
): { top: number; left: number } {
  const maxLeft = window.innerWidth - OVERFLOW_POPOVER_GUTTER_PX - popoverWidth;
  let left = trigger.left;
  if (left > maxLeft) {
    left = trigger.right - popoverWidth;
  }
  left = Math.min(
    Math.max(OVERFLOW_POPOVER_GUTTER_PX, left),
    Math.max(OVERFLOW_POPOVER_GUTTER_PX, maxLeft),
  );
  return { top: trigger.bottom + OVERFLOW_POPOVER_OFFSET_PX, left };
}

export interface SkillCardPill {
  id: string;
  node: React.ReactNode;
}

function computeVisiblePillCount(
  widths: number[],
  containerWidth: number,
): number {
  if (widths.length === 0 || containerWidth <= 0) return 0;

  let used = 0;
  for (let i = 0; i < widths.length; i += 1) {
    const width = widths[i]!;
    const gap = i > 0 ? PILL_GAP_PX : 0;
    const remaining = widths.length - i - 1;
    const reserve = remaining > 0 ? OVERFLOW_PILL_WIDTH_PX + PILL_GAP_PX : 0;
    if (used + gap + width + reserve > containerWidth) {
      return Math.max(1, i);
    }
    used += gap + width;
  }
  return widths.length;
}

interface SkillCardPillRowProps {
  pills: SkillCardPill[];
  testId: string;
}

export function SkillCardPillRow({ pills, testId }: SkillCardPillRowProps) {
  const { t } = useTranslation("openhands");
  const containerRef = React.useRef<HTMLDivElement>(null);
  const measureRef = React.useRef<HTMLDivElement>(null);
  const triggerRef = React.useRef<HTMLButtonElement>(null);
  const popoverRef = React.useRef<HTMLDivElement>(null);
  const [visibleCount, setVisibleCount] = React.useState(pills.length);
  const [isOverflowOpen, setIsOverflowOpen] = React.useState(false);
  const [popoverBox, setPopoverBox] = React.useState<{
    top: number;
    left: number;
  } | null>(null);

  const recomputeVisibleCount = React.useCallback(() => {
    const container = containerRef.current;
    const measure = measureRef.current;
    if (!container || !measure) return;

    const widths = Array.from(measure.children).map(
      (child) => (child as HTMLElement).offsetWidth,
    );
    setVisibleCount(computeVisiblePillCount(widths, container.clientWidth));
  }, []);

  // Recommended cards rebuild the pills array every render; compare ids so an
  // open +N popover is not slammed shut under the cursor.
  const pillsKey = pills.map((pill) => pill.id).join("\u001f");
  const lastPillsKeyRef = React.useRef<string | null>(null);

  React.useLayoutEffect(() => {
    if (lastPillsKeyRef.current === pillsKey) {
      return;
    }
    lastPillsKeyRef.current = pillsKey;
    setIsOverflowOpen(false);
    recomputeVisibleCount();
  }, [pillsKey, recomputeVisibleCount]);

  React.useEffect(() => {
    const container = containerRef.current;
    if (!container || typeof ResizeObserver === "undefined") return undefined;

    const observer = new ResizeObserver(() => recomputeVisibleCount());
    observer.observe(container);
    return () => observer.disconnect();
  }, [recomputeVisibleCount]);

  const measurePopover = React.useCallback(() => {
    const trigger = triggerRef.current;
    if (!trigger) {
      return;
    }
    const next = placePopoverByTrigger(
      trigger.getBoundingClientRect(),
      popoverRef.current?.offsetWidth ?? trigger.offsetWidth,
    );
    setPopoverBox((prev) =>
      prev?.top === next.top && prev?.left === next.left ? prev : next,
    );
  }, []);

  React.useLayoutEffect(() => {
    if (!isOverflowOpen) {
      setPopoverBox(null);
      return undefined;
    }
    measurePopover();
    window.addEventListener("resize", measurePopover);
    window.addEventListener("scroll", measurePopover, true);
    return () => {
      window.removeEventListener("resize", measurePopover);
      window.removeEventListener("scroll", measurePopover, true);
    };
  }, [isOverflowOpen, measurePopover]);

  React.useLayoutEffect(() => {
    if (isOverflowOpen && popoverBox && popoverRef.current) {
      measurePopover();
    }
  }, [isOverflowOpen, measurePopover, popoverBox]);

  React.useEffect(() => {
    if (!isOverflowOpen) {
      return undefined;
    }
    const onPointerDown = (event: MouseEvent) => {
      const target = event.target as Node;
      if (triggerRef.current?.contains(target)) {
        return;
      }
      if (popoverRef.current?.contains(target)) {
        return;
      }
      setIsOverflowOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsOverflowOpen(false);
      }
    };
    // mousedown (not click) so the opening click cannot race-close the panel,
    // and so wrapping card/link activation is easier to cancel on the trigger.
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [isOverflowOpen]);

  if (pills.length === 0) return null;

  const hiddenCount = Math.max(0, pills.length - visibleCount);
  const overflowPills = pills.slice(visibleCount);

  const stopCardActivation = (event: React.SyntheticEvent) => {
    event.stopPropagation();
  };

  const activateOverflow = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsOverflowOpen((open) => !open);
  };

  return (
    <div data-testid={`${testId}-wrap`} className="min-w-0 overflow-hidden">
      <div
        ref={measureRef}
        aria-hidden
        className="pointer-events-none fixed top-0 -left-[10000px] z-[-1] flex flex-nowrap items-center gap-1.5 opacity-0"
      >
        {pills.map((pill) => (
          <span key={pill.id} className="inline-flex shrink-0">
            {pill.node}
          </span>
        ))}
      </div>
      <div
        ref={containerRef}
        data-testid={testId}
        className="flex w-full min-w-0 max-w-full flex-nowrap items-center gap-1.5 overflow-hidden"
      >
        {pills.slice(0, visibleCount).map((pill) => (
          <span key={pill.id} className="inline-flex shrink-0">
            {pill.node}
          </span>
        ))}
        {hiddenCount > 0 ? (
          <button
            ref={triggerRef}
            type="button"
            data-testid={`${testId}-overflow`}
            aria-expanded={isOverflowOpen}
            aria-haspopup="dialog"
            aria-label={t(I18nKey.SETTINGS$SKILLS_PILLS_OVERFLOW_ARIA, {
              count: hiddenCount,
            })}
            onMouseDown={stopCardActivation}
            onClick={activateOverflow}
            onKeyDown={stopCardActivation}
            className={cn(
              extensionModuleCardPillClassName,
              "cursor-pointer font-medium text-tertiary-alt hover:text-white",
            )}
          >
            {t(I18nKey.SETTINGS$SKILLS_PILLS_MORE, { count: hiddenCount })}
          </button>
        ) : null}
      </div>

      {isOverflowOpen &&
        popoverBox &&
        typeof document !== "undefined" &&
        createPortal(
          // Stop card-level click activation when interacting with the list.
          // eslint-disable-next-line jsx-a11y/no-noninteractive-element-interactions -- dialog surface must swallow clicks
          <div
            ref={popoverRef}
            role="dialog"
            data-testid={`${testId}-overflow-popover`}
            className={cn(
              "z-[9999] flex w-max max-w-[20rem] flex-col gap-1.5",
              "rounded-md border border-[var(--oh-border-subtle)]",
              "bg-[var(--oh-surface)] p-2 shadow-lg",
            )}
            style={{
              position: "fixed",
              top: popoverBox.top,
              left: popoverBox.left,
            }}
            onMouseDown={stopCardActivation}
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
            }}
          >
            {overflowPills.map((pill) => (
              <div
                key={pill.id}
                data-testid={`${testId}-overflow-item`}
                className="min-w-0"
              >
                {pill.node}
              </div>
            ))}
          </div>,
          document.body,
        )}
    </div>
  );
}
