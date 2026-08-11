import { ExternalLink } from "lucide-react";
import { SiSlack } from "react-icons/si";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { OPENHANDS_SLACK_INVITE_URL } from "#/utils/constants";
import { cn } from "#/utils/utils";
import {
  SIDEBAR_ICON_SLOT_CLASS,
  SIDEBAR_ROW_INTERACTIVE_CLASS,
  sidebarNavLabelClassName,
  sidebarNavRowClassName,
} from "#/components/features/sidebar/sidebar-layout";

export function JoinSlackSettingsLink() {
  const { t } = useTranslation("openhands");

  return (
    <a
      data-testid="settings-join-slack-link"
      href={OPENHANDS_SLACK_INVITE_URL}
      target="_blank"
      rel="noopener noreferrer"
      className={cn(
        sidebarNavRowClassName({ collapsed: false }),
        SIDEBAR_ROW_INTERACTIVE_CLASS.idle,
      )}
    >
      <span className={SIDEBAR_ICON_SLOT_CLASS}>
        <SiSlack className="size-4 shrink-0" aria-hidden />
      </span>
      <span className={cn(sidebarNavLabelClassName(false), "flex-1")}>
        {t(I18nKey.SIDEBAR$JOIN_SLACK)}
      </span>
      <ExternalLink
        className="size-4 shrink-0 text-[var(--oh-muted)]"
        aria-hidden
      />
    </a>
  );
}
