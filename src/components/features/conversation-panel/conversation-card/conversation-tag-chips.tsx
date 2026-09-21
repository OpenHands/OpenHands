import React from "react";
import { createPortal } from "react-dom";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import {
  computeVisibleTagChipCount,
  formatConversationTagTooltip,
  getConversationTagLabel,
  truncateTagChipValue,
} from "./conversation-tag-display";
import {
  getConversationTagIcon,
  type ConversationTagIcon,
} from "./conversation-tag-icons";
import {
  CONVERSATION_CARD_META_CHIP_CLASSNAME,
  CONVERSATION_CARD_META_CHIP_ICON_CLASSNAME,
  CONVERSATION_CARD_META_CHIP_ICON_SLOT_CLASSNAME,
} from "./conversation-card-meta-chip";

interface ConversationTagChipsProps {
  tags: Array<[string, string]>;
}

/**
 * Fixed line-height slot so Lucide / react-icons / local SVGs share the same
 * optical center as the ``leading-4`` chip / overflow text (raw ``h-3 w-3``
 * SVGs sit on the baseline and look high or low depending on the glyph).
 */
function TagIconSlot({
  icon: Icon,
  keyName,
  testId,
}: {
  icon: ConversationTagIcon;
  keyName?: string;
  testId?: string;
}) {
  return (
    <span
      className={CONVERSATION_CARD_META_CHIP_ICON_SLOT_CLASSNAME}
      aria-hidden
    >
      <Icon
        data-testid={testId}
        data-tag-key={keyName}
        aria-hidden
        className={CONVERSATION_CARD_META_CHIP_ICON_CLASSNAME}
      />
    </span>
  );
}

function TagChipContent({
  icon,
  keyName,
  value,
  iconTestId,
}: {
  icon: ConversationTagIcon;
  keyName: string;
  value: string;
  /**
   * Only the visible row passes this. The off-screen measure row renders the
   * same chips, so tagging both would emit every chip test id twice and break
   * ``getByTestId`` (singular) for any card with tags.
   */
  iconTestId?: string;
}) {
  return (
    <>
      <TagIconSlot icon={icon} keyName={keyName} testId={iconTestId} />
      {/* Bare tags (empty value) fall back to the key so the chip is never
          an empty pill. */}
      <span className="truncate leading-4">
        {truncateTagChipValue(value || keyName)}
      </span>
    </>
  );
}

/**
 * Single-row tag chips for a conversation card. Chip labels are value-only
 * (bare tags with an empty value show the key; the full ``key: value`` pair
 * lives in the tooltip); chips that do not fit fold behind a ``+N``
 * button that opens a key/value popover.
 *
 * The overflow popover is portaled with ``position: fixed`` so it is not
 * clipped by the chip row's ``overflow-hidden`` or the sidebar scroller.
 */
export function ConversationTagChips({ tags }: ConversationTagChipsProps) {
  const { t } = useTranslation("openhands");
  const containerRef = React.useRef<HTMLDivElement>(null);
  const measureRef = React.useRef<HTMLDivElement>(null);
  const triggerRef = React.useRef<HTMLButtonElement>(null);
  const popoverRef = React.useRef<HTMLDivElement>(null);
  const [visibleCount, setVisibleCount] = React.useState(tags.length);
  const [isOverflowOpen, setIsOverflowOpen] = React.useState(false);
  const [popoverBox, setPopoverBox] = React.useState<{
    top: number;
    left: number;
  } | null>(null);

  const recomputeVisibleCount = React.useCallback(() => {
    const container = containerRef.current;
    const measure = measureRef.current;
    if (!container || !measure) {
      return;
    }

    const widths = Array.from(measure.children).map(
      (child) => (child as HTMLElement).offsetWidth,
    );
    setVisibleCount(computeVisibleTagChipCount(widths, container.clientWidth));
  }, []);

  const measurePopover = React.useCallback(() => {
    const trigger = triggerRef.current;
    if (!trigger) {
      return;
    }
    const rect = trigger.getBoundingClientRect();
    const gutter = 8;
    const width = 16 * 16;
    let left = rect.left;
    if (left + width > window.innerWidth - gutter) {
      left = Math.max(gutter, window.innerWidth - gutter - width);
    }
    setPopoverBox({ top: rect.bottom + 4, left });
  }, []);

  // Callers build this array fresh on every render (``getDisplayConversationTags``
  // returns a new array), so keying the reset effect on ``tags`` identity would
  // fire on every parent render — closing an open ``+N`` popover and forcing a
  // layout read on each 10s panel refetch. Compare tag *content* instead.
  const tagsKey = tags.map(([key, value]) => `${key}=${value}`).join("\u001f");
  const lastTagsKeyRef = React.useRef<string | null>(null);

  React.useLayoutEffect(() => {
    if (lastTagsKeyRef.current === tagsKey) {
      return;
    }
    lastTagsKeyRef.current = tagsKey;
    setIsOverflowOpen(false);
    recomputeVisibleCount();
  }, [tagsKey, recomputeVisibleCount]);

  React.useEffect(() => {
    const container = containerRef.current;
    if (!container || typeof ResizeObserver === "undefined") {
      return undefined;
    }

    const observer = new ResizeObserver(() => recomputeVisibleCount());
    observer.observe(container);
    return () => observer.disconnect();
  }, [recomputeVisibleCount]);

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
    // and so NavigationLink activation is easier to cancel on the trigger.
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [isOverflowOpen]);

  if (tags.length === 0) {
    return null;
  }

  const hiddenCount = Math.max(0, tags.length - visibleCount);
  const visibleTags = tags.slice(0, visibleCount);
  const overflowTags = tags.slice(visibleCount);

  const stopCardNavigation = (event: React.SyntheticEvent) => {
    // Don't preventDefault on mousedown — that suppresses the subsequent click
    // when the chip sits inside a conversation NavigationLink.
    event.stopPropagation();
  };

  const activateOverflow = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsOverflowOpen((open) => !open);
  };

  return (
    <div
      data-testid="conversation-card-tag-chips"
      className="relative min-w-0 w-full"
    >
      <div
        ref={measureRef}
        aria-hidden
        className="pointer-events-none fixed top-0 -left-[10000px] z-[-1] flex flex-nowrap items-center gap-1 opacity-0"
      >
        {tags.map(([key, value]) => (
          <span key={key} className={CONVERSATION_CARD_META_CHIP_CLASSNAME}>
            <TagChipContent
              icon={getConversationTagIcon(key, value)}
              keyName={key}
              value={value}
            />
          </span>
        ))}
      </div>

      <div
        ref={containerRef}
        data-testid="conversation-card-tag-row"
        className="flex w-full min-w-0 max-w-full flex-nowrap items-center gap-1 overflow-hidden"
      >
        {visibleTags.map(([key, value]) => (
          <span
            key={key}
            data-testid="conversation-card-tag-chip"
            title={formatConversationTagTooltip(key, value, t)}
            className={CONVERSATION_CARD_META_CHIP_CLASSNAME}
          >
            <TagChipContent
              icon={getConversationTagIcon(key, value)}
              keyName={key}
              value={value}
              iconTestId="conversation-card-tag-chip-icon"
            />
          </span>
        ))}
        {hiddenCount > 0 ? (
          <button
            ref={triggerRef}
            type="button"
            data-testid="conversation-card-tag-overflow"
            aria-expanded={isOverflowOpen}
            aria-haspopup="dialog"
            aria-label={t(I18nKey.CONVERSATION$TAGS_OVERFLOW_ARIA, {
              count: hiddenCount,
            })}
            onMouseDown={stopCardNavigation}
            onClick={activateOverflow}
            className={cn(
              CONVERSATION_CARD_META_CHIP_CLASSNAME,
              "shrink-0 self-center hover:bg-[var(--oh-interactive-hover)] hover:text-[var(--foreground)]",
            )}
          >
            {/* Match chip height (icon slot + label) so +N shares the same centerline. */}
            <span className="inline-flex h-4 w-0 shrink-0" aria-hidden />
            <span className="leading-4">{`+${hiddenCount}`}</span>
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
            data-testid="conversation-card-tag-overflow-popover"
            className={cn(
              "z-[9999] min-w-[10rem] max-w-[16rem]",
              "rounded-md border border-[var(--oh-border-subtle)]",
              "bg-[var(--oh-surface)] p-2 shadow-lg",
            )}
            style={{
              position: "fixed",
              top: popoverBox.top,
              left: popoverBox.left,
            }}
            onMouseDown={stopCardNavigation}
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
            }}
          >
            <dl className="m-0 flex flex-col gap-1">
              {overflowTags.map(([key, value]) => (
                <div
                  key={key}
                  data-testid="conversation-card-tag-overflow-row"
                  className="grid grid-cols-[minmax(0,auto)_minmax(0,1fr)] gap-x-2 text-[10px] leading-4"
                >
                  <dt className="m-0 inline-flex min-w-0 items-center gap-1 whitespace-normal break-words text-[var(--oh-muted)]">
                    <TagIconSlot icon={getConversationTagIcon(key, value)} />
                    <span className="min-w-0 whitespace-normal break-words leading-4">
                      {getConversationTagLabel(key, t)}
                    </span>
                  </dt>
                  <dd
                    className="m-0 whitespace-normal break-words text-left leading-4 text-[var(--foreground)]"
                    title={value}
                  >
                    {value}
                  </dd>
                </div>
              ))}
            </dl>
          </div>,
          document.body,
        )}
    </div>
  );
}
