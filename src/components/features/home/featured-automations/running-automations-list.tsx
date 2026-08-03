import { Pin } from "lucide-react";
import { useLayoutEffect, useMemo, useReducer, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useTranslation } from "react-i18next";
import { RunStatusBadge } from "#/components/features/automations/detail/run-status-badge";
import { ContextMenuListItem } from "#/components/features/context-menu/context-menu-list-item";
import { ConversationNameContextMenuIconText } from "#/components/features/conversation/conversation-name-context-menu-icon-text";
import { EllipsisButton } from "#/components/features/conversation-panel/ellipsis-button";
import { NavigationLink } from "#/components/shared/navigation-link";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import {
  UNKNOWN_RUN_STATE,
  useHomeAutomations,
} from "#/hooks/query/use-home-automations";
import { useHomePinnedAutomations } from "#/hooks/use-home-pinned-automations";
import { I18nKey } from "#/i18n/declaration";
import { ContextMenu } from "#/ui/context-menu";
import {
  buildHomeAutomationActivityItems,
  hrefForActivityItem,
  type HomeAutomationActivityItem,
} from "./home-automation-activity";

interface RunningAutomationRowProps {
  item: HomeAutomationActivityItem;
}

function RunningAutomationRow({ item }: RunningAutomationRowProps) {
  const { t } = useTranslation("openhands");
  const { isPinned, togglePin } = useHomePinnedAutomations();
  const [menuOpen, setMenuOpen] = useState(false);
  const pinned = isPinned(item.id);
  const anchorRef = useRef<HTMLButtonElement>(null);
  const [, bumpPosition] = useReducer((i: number) => i + 1, 0);
  const menuRef = useClickOutsideElement<HTMLUListElement>(
    () => setMenuOpen(false),
    anchorRef,
  );

  useLayoutEffect(() => {
    if (!menuOpen) return undefined;
    bumpPosition();
    let frame = 0;
    const update = () => {
      if (frame) return;
      frame = window.requestAnimationFrame(() => {
        frame = 0;
        bumpPosition();
      });
    };
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    return () => {
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, [menuOpen]);

  const floatingStyle = (() => {
    if (!menuOpen || !anchorRef.current) return undefined;
    const rect = anchorRef.current.getBoundingClientRect();
    const gutter = 8;
    const vw = window.innerWidth;
    return {
      position: "fixed" as const,
      top: rect.bottom + 4,
      right: Math.max(gutter, vw - rect.right),
      zIndex: 100_000,
    };
  })();

  const portalTarget = typeof document !== "undefined" ? document.body : null;
  const metaLine = [item.triggerSummary, item.whenLabel]
    .filter(Boolean)
    .join(" · ");

  return (
    <li
      data-testid={`running-automation-row-${item.id}`}
      className="group relative flex items-stretch"
    >
      <NavigationLink
        to={hrefForActivityItem(item)}
        aria-label={item.name}
        className="flex min-w-0 flex-1 items-center justify-between gap-3 px-3 py-2.5 transition-colors hover:bg-[var(--oh-interactive-hover)] focus:outline-none focus-visible:bg-[var(--oh-interactive-hover)] focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[var(--oh-focus)]"
      >
        <div className="min-w-0 flex-1">
          <span className="block truncate text-sm font-medium text-[var(--oh-foreground)]">
            {item.name}
          </span>
          {metaLine ? (
            <span className="mt-0.5 block truncate text-xs text-[var(--oh-text-secondary)]">
              {metaLine}
            </span>
          ) : null}
        </div>

        {item.status ? <RunStatusBadge status={item.status} /> : null}
      </NavigationLink>

      <div className="relative flex shrink-0 items-center pr-1.5">
        <EllipsisButton
          ref={anchorRef}
          testId={`running-automation-menu-${item.id}`}
          ariaLabel={t(I18nKey.FEATURED_AUTOMATIONS$ROW_MENU_LABEL, {
            name: item.name,
          })}
          className="opacity-70 group-hover:opacity-100"
          onClick={(event) => {
            event.preventDefault();
            event.stopPropagation();
            setMenuOpen((open) => !open);
          }}
        />

        {menuOpen && floatingStyle && portalTarget
          ? createPortal(
              <ContextMenu
                ref={menuRef}
                testId={`running-automation-menu-panel-${item.id}`}
                theme="popover"
                position="none"
                alignment="none"
                spacing="none"
                style={floatingStyle}
                className="min-w-40"
              >
                <ContextMenuListItem
                  testId={`running-automation-pin-${item.id}`}
                  onClick={(event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    togglePin(item.id);
                    setMenuOpen(false);
                  }}
                >
                  <ConversationNameContextMenuIconText
                    icon={
                      <Pin
                        className={
                          pinned ? "size-3.5 fill-current" : "size-3.5"
                        }
                        aria-hidden="true"
                      />
                    }
                    text={t(
                      pinned
                        ? I18nKey.FEATURED_AUTOMATIONS$UNPIN
                        : I18nKey.FEATURED_AUTOMATIONS$PIN,
                    )}
                  />
                </ContextMenuListItem>
              </ContextMenu>,
              portalTarget,
            )
          : null}
      </div>
    </li>
  );
}

/**
 * Recent automation activity under the home composer. Driven by live
 * enabled automations + latest-run queries; self-gates when the automation
 * service is unavailable or there are no enabled automations.
 */
export function RunningAutomationsList() {
  const { t, i18n } = useTranslation("openhands");
  const {
    isBackendHealthy,
    isHealthLoading,
    isError,
    enabledAutomations,
    runStates,
  } = useHomeAutomations();

  const items = useMemo(
    () =>
      buildHomeAutomationActivityItems(
        enabledAutomations,
        runStates,
        i18n.language,
        t,
        UNKNOWN_RUN_STATE,
      ),
    [enabledAutomations, runStates, i18n.language, t],
  );

  if (isHealthLoading || !isBackendHealthy || isError || items.length === 0) {
    return null;
  }

  return (
    <section
      aria-labelledby="running-automations-heading"
      data-testid="running-automations-list"
      className="w-full"
    >
      <h2
        id="running-automations-heading"
        className="mb-2 text-sm font-medium text-[var(--oh-foreground)]"
      >
        {t(I18nKey.FEATURED_AUTOMATIONS$RECENT_TITLE)}
      </h2>

      <ul
        aria-label={t(I18nKey.FEATURED_AUTOMATIONS$RECENT_GROUP_LABEL)}
        className="divide-y divide-[var(--oh-border-subtle)] overflow-hidden rounded-xl border border-[var(--oh-border-subtle)] bg-[var(--oh-surface)]"
      >
        {items.map((item) => (
          <RunningAutomationRow key={item.id} item={item} />
        ))}
      </ul>
    </section>
  );
}
