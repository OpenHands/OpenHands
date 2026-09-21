import { useRef, useState } from "react";
import { AlertTriangle } from "lucide-react";
import { useTranslation } from "react-i18next";
import { AgentProfileActionsMenu } from "./agent-profile-actions-menu";
import { type AgentProfileSummary } from "#/api/agent-profiles-service/agent-profiles-service.api";
import { I18nKey } from "#/i18n/declaration";
import { EllipsisButton } from "#/components/features/conversation-panel/ellipsis-button";
import { BrandBadge } from "#/components/shared/badge";
import { cn } from "#/utils/utils";
import {
  settingsListIconActionButtonClassName,
  settingsListRowClassName,
  settingsListWarningBadgeClassName,
} from "#/utils/settings-list-classes";

interface AgentProfileRowProps {
  profile: AgentProfileSummary;
  isActive: boolean;
  /** When false, the row is read-only and the actions menu is hidden. */
  canManage: boolean;
  /**
   * The LLM profile a launch from this profile will really use, when it has
   * drifted from `llm_profile_ref` — see {@link getAgentProfileLlmDrift}.
   * `null` (the common case) hides the warning badge.
   */
  driftLlmProfile: string | null;
  onActivate: (profile: AgentProfileSummary) => void;
  onEdit: (profile: AgentProfileSummary) => void;
  onDelete: (profile: AgentProfileSummary) => void;
  isActivating: boolean;
}

export function AgentProfileRow({
  profile,
  isActive,
  canManage,
  driftLlmProfile,
  onActivate,
  onEdit,
  onDelete,
  isActivating,
}: AgentProfileRowProps) {
  const { t } = useTranslation("openhands");
  const [menuOpen, setMenuOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);

  // Secondary label: the referenced LLM profile (OpenHands) or the "ACP" kind.
  const secondary =
    profile.agent_kind === "openhands"
      ? profile.llm_profile_ref
      : t(I18nKey.SETTINGS$AGENT_TYPE_ACP);

  return (
    <div
      data-testid="agent-profile-row"
      className={cn(settingsListRowClassName, "justify-between gap-3")}
    >
      <div className="flex min-w-0 flex-1 items-center gap-3">
        <span
          className="min-w-0 max-w-full truncate text-sm font-medium text-white"
          title={profile.name}
        >
          {profile.name}
        </span>
        {secondary ? (
          <span
            className="min-w-0 max-w-full truncate text-sm text-[var(--oh-muted)]"
            title={secondary}
          >
            {secondary}
          </span>
        ) : null}
        {isActive && (
          <BrandBadge
            className="shrink-0 whitespace-nowrap px-2.5 py-1 text-xs"
            data-testid="agent-profile-active-badge"
          >
            {t(I18nKey.SETTINGS$PROFILE_DEFAULT)}
          </BrandBadge>
        )}
        {driftLlmProfile && (
          <span
            className={settingsListWarningBadgeClassName}
            title={t(I18nKey.SETTINGS$AGENT_PROFILE_LLM_DRIFT_TOOLTIP, {
              active: driftLlmProfile,
              pinned: profile.llm_profile_ref,
            })}
            data-testid="agent-profile-llm-drift-badge"
          >
            <AlertTriangle className="h-3 w-3" aria-hidden />
            {t(I18nKey.SETTINGS$AGENT_PROFILE_LLM_DRIFT)}
          </span>
        )}
      </div>
      {canManage && (
        <div className="relative shrink-0">
          <EllipsisButton
            ref={triggerRef}
            onClick={() => setMenuOpen((open) => !open)}
            ariaLabel={t(I18nKey.SETTINGS$PROFILE_MENU)}
            testId="agent-profile-menu-trigger"
            className={settingsListIconActionButtonClassName}
          />
          {menuOpen && (
            <AgentProfileActionsMenu
              anchorRef={triggerRef}
              onEdit={() => onEdit(profile)}
              onSetActive={() => onActivate(profile)}
              onDelete={() => onDelete(profile)}
              isActive={isActive}
              isActivating={isActivating}
              onClose={() => setMenuOpen(false)}
            />
          )}
        </div>
      )}
    </div>
  );
}
