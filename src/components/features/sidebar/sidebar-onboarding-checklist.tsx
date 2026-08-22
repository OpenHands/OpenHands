import { Tooltip } from "@heroui/react";
import { Check, ChevronDown } from "lucide-react";
import { useTranslation } from "react-i18next";
import { NavigationLink } from "#/components/shared/navigation-link";
import { useTracking } from "#/hooks/use-tracking";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import {
  getSidebarOnboardingChecklistHref,
  isExternalSidebarOnboardingChecklistItem,
  SIDEBAR_ONBOARDING_CHECKLIST_DESTINATION_TYPES,
  SIDEBAR_ONBOARDING_CHECKLIST_I18N_KEYS,
  SIDEBAR_ONBOARDING_CHECKLIST_LINK_IDS,
  type SidebarOnboardingChecklistItemId,
} from "./sidebar-onboarding-checklist.constants";
import { SidebarOnboardingChecklistItemPreview } from "./sidebar-onboarding-checklist-item-preview";
import { useSidebarOnboardingChecklist } from "./use-sidebar-onboarding-checklist";

const CHECKLIST_ITEM_TOOLTIP_CLASS =
  "rounded-xl border border-[var(--oh-border)] bg-base-secondary p-0 text-white shadow-xl";

const CHECKLIST_ITEM_CLASS = cn(
  "flex min-w-0 w-full items-center gap-2.5 rounded-md px-2.5 py-1.5 text-sm",
  "transition-colors hover:bg-[var(--oh-surface)]",
);

interface SidebarOnboardingChecklistProps {
  collapsed: boolean;
}

function ChecklistStatusIcon({ isComplete }: { isComplete: boolean }) {
  return (
    <span
      aria-hidden
      className={cn(
        "inline-flex size-4 shrink-0 items-center justify-center rounded-full border",
        isComplete
          ? "border-primary bg-primary text-[var(--oh-color-base)]"
          : "border-[var(--oh-border)] bg-transparent",
      )}
    >
      {isComplete ? <Check className="size-2.5" strokeWidth={3} /> : null}
    </span>
  );
}

function ChecklistItem({
  id,
  isComplete,
  onActivate,
}: {
  id: SidebarOnboardingChecklistItemId;
  isComplete: boolean;
  onActivate?: () => void;
}) {
  const { t } = useTranslation("openhands");
  const { trackOnboardingLinkClicked } = useTracking();
  const labelKey = SIDEBAR_ONBOARDING_CHECKLIST_I18N_KEYS[id];
  const disableAnimation = import.meta.env.MODE === "test";
  const href = getSidebarOnboardingChecklistHref(id);
  const linkId = SIDEBAR_ONBOARDING_CHECKLIST_LINK_IDS[id];

  const handleLinkClick = () => {
    trackOnboardingLinkClicked({
      linkId,
      destinationType: SIDEBAR_ONBOARDING_CHECKLIST_DESTINATION_TYPES[id],
      surface: "landing_checklist",
      checklistItem: linkId,
      isExternal: href.kind === "external",
    });
    onActivate?.();
  };

  const handleDocsClick = () => {
    trackOnboardingLinkClicked({
      linkId: "open_docs",
      destinationType: "documentation",
      surface: "landing_checklist",
      checklistItem: linkId,
      isExternal: true,
    });
  };

  const itemClassName = cn(
    CHECKLIST_ITEM_CLASS,
    isComplete ? "text-muted" : "text-content",
  );
  const label = (
    <>
      <ChecklistStatusIcon isComplete={isComplete} />
      <span
        className={cn("min-w-0 flex-1 truncate", isComplete && "line-through")}
      >
        {t(labelKey)}
      </span>
    </>
  );

  return (
    <li>
      <Tooltip
        placement="right-start"
        delay={0}
        closeDelay={100}
        disableAnimation={disableAnimation}
        className={CHECKLIST_ITEM_TOOLTIP_CLASS}
        content={
          <SidebarOnboardingChecklistItemPreview
            id={id}
            onActionClick={handleLinkClick}
            onDocsClick={handleDocsClick}
          />
        }
      >
        {href.kind === "external" ? (
          <a
            href={href.href}
            target="_blank"
            rel="noreferrer"
            data-testid={`sidebar-onboarding-checklist-item-${id}`}
            className={itemClassName}
            onClick={handleLinkClick}
          >
            {label}
          </a>
        ) : (
          <NavigationLink
            to={href.href}
            data-testid={`sidebar-onboarding-checklist-item-${id}`}
            className={itemClassName}
            onClick={handleLinkClick}
          >
            {label}
          </NavigationLink>
        )}
      </Tooltip>
    </li>
  );
}

export function SidebarOnboardingChecklist({
  collapsed,
}: SidebarOnboardingChecklistProps) {
  const { t } = useTranslation("openhands");
  const {
    items,
    completedCount,
    isVisible,
    isMinimized,
    toggleMinimized,
    markJoinSlackComplete,
  } = useSidebarOnboardingChecklist();

  if (collapsed || !isVisible) {
    return null;
  }

  return (
    <div
      data-testid="sidebar-onboarding-checklist"
      data-minimized={isMinimized ? "true" : "false"}
      className={cn(
        "w-full shrink-0 overflow-hidden rounded-xl border border-[var(--oh-border)]",
        "bg-[var(--oh-surface-raised)] shadow-sm",
      )}
    >
      <div className={cn("px-1", isMinimized ? "py-1" : "pt-2 pb-1")}>
        <div
          className={cn(
            "relative flex w-full items-center gap-0.5 rounded-md px-1.5",
            "transition-colors hover:bg-[var(--oh-surface)]",
            isMinimized ? "py-1" : "py-1.5",
          )}
        >
          <button
            type="button"
            data-testid="sidebar-onboarding-checklist-toggle"
            aria-expanded={!isMinimized}
            aria-label={
              isMinimized
                ? t(I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_EXPAND)
                : t(I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_COLLAPSE)
            }
            onClick={toggleMinimized}
            className="absolute inset-0 z-0 cursor-pointer rounded-md"
          />

          <div className="relative z-10 min-w-0 flex-1 px-0.5 pointer-events-none">
            <div className="flex min-w-0 flex-wrap items-baseline gap-x-2 gap-y-0.5">
              <span className="text-sm font-semibold text-content">
                {t(I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_TITLE)}
              </span>
              <span className="text-xs text-muted">
                {t(I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_PROGRESS, {
                  completed: completedCount,
                })}
              </span>
            </div>
          </div>

          <span
            data-testid="sidebar-onboarding-checklist-chevron"
            aria-hidden
            className={cn(
              "relative z-10 inline-flex size-7 shrink-0 items-center justify-center",
              "pointer-events-none text-[var(--oh-muted)]",
            )}
          >
            <ChevronDown
              className={cn(
                "size-4 transition-transform motion-reduce:transition-none",
                isMinimized && "-rotate-90",
              )}
            />
          </span>
        </div>
      </div>

      {!isMinimized ? (
        <ul className="flex flex-col gap-0.5 px-2.5 pb-2">
          {items.map((item) => (
            <ChecklistItem
              key={item.id}
              id={item.id}
              isComplete={item.isComplete}
              onActivate={
                isExternalSidebarOnboardingChecklistItem(item.id)
                  ? markJoinSlackComplete
                  : undefined
              }
            />
          ))}
        </ul>
      ) : null}
    </div>
  );
}
