import { BookOpen } from "lucide-react";
import { useTranslation } from "react-i18next";
import { NavigationLink } from "#/components/shared/navigation-link";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import {
  getSidebarOnboardingChecklistHref,
  SIDEBAR_ONBOARDING_CHECKLIST_ACTION_I18N_KEYS,
  SIDEBAR_ONBOARDING_CHECKLIST_DESCRIPTION_I18N_KEYS,
  SIDEBAR_ONBOARDING_CHECKLIST_DOCS_URLS,
  SIDEBAR_ONBOARDING_CHECKLIST_I18N_KEYS,
  type SidebarOnboardingChecklistItemId,
} from "./sidebar-onboarding-checklist.constants";
import { SidebarOnboardingChecklistItemIcon } from "./sidebar-onboarding-checklist-item-icon";

const PREVIEW_ACTION_BUTTON_CLASS = cn(
  "inline-flex shrink-0 items-center rounded-md bg-white px-2.5 py-1",
  "text-xs font-medium text-black transition-colors hover:bg-white/90",
);

interface SidebarOnboardingChecklistItemPreviewProps {
  id: SidebarOnboardingChecklistItemId;
  onActionClick?: () => void;
  onDocsClick?: () => void;
}

export function SidebarOnboardingChecklistItemPreview({
  id,
  onActionClick,
  onDocsClick,
}: SidebarOnboardingChecklistItemPreviewProps) {
  const { t } = useTranslation("openhands");
  const titleKey = SIDEBAR_ONBOARDING_CHECKLIST_I18N_KEYS[id];
  const descriptionKey = SIDEBAR_ONBOARDING_CHECKLIST_DESCRIPTION_I18N_KEYS[id];
  const actionKey = SIDEBAR_ONBOARDING_CHECKLIST_ACTION_I18N_KEYS[id];
  const docsUrl = SIDEBAR_ONBOARDING_CHECKLIST_DOCS_URLS[id];
  const href = getSidebarOnboardingChecklistHref(id);

  return (
    <div
      className="flex w-[280px] flex-col gap-2.5 p-3"
      data-testid={`sidebar-onboarding-checklist-preview-${id}`}
    >
      <div className="flex items-start gap-2">
        <span className="mt-0.5 inline-flex shrink-0 items-center justify-center">
          <SidebarOnboardingChecklistItemIcon id={id} />
        </span>
        <span className="min-w-0 flex-1 text-sm font-medium text-white">
          {t(titleKey)}
        </span>
      </div>
      <p className="text-xs leading-relaxed text-[var(--oh-muted)]">
        {t(descriptionKey)}
      </p>
      <div className="flex items-center justify-between gap-3">
        <a
          href={docsUrl}
          target="_blank"
          rel="noreferrer"
          data-testid={`sidebar-onboarding-checklist-preview-docs-${id}`}
          className="inline-flex min-w-0 items-center gap-2 text-xs text-[var(--oh-muted)] transition-colors hover:text-white hover:underline"
          onClick={onDocsClick}
        >
          <BookOpen className="size-3.5 shrink-0" aria-hidden />
          {t(I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_DOCS_LINK)}
        </a>
        {href.kind === "external" ? (
          <a
            href={href.href}
            target="_blank"
            rel="noreferrer"
            data-testid={`sidebar-onboarding-checklist-preview-action-${id}`}
            className={PREVIEW_ACTION_BUTTON_CLASS}
            onClick={onActionClick}
          >
            {t(actionKey)}
          </a>
        ) : (
          <NavigationLink
            to={href.href}
            data-testid={`sidebar-onboarding-checklist-preview-action-${id}`}
            className={PREVIEW_ACTION_BUTTON_CLASS}
            onClick={onActionClick}
          >
            {t(actionKey)}
          </NavigationLink>
        )}
      </div>
    </div>
  );
}
