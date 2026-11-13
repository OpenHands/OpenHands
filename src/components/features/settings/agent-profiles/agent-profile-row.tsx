import { useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { AgentProfileActionsMenu } from "./agent-profile-actions-menu";
import {
  WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME,
  type AgentProfileSummary,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { I18nKey } from "#/i18n/declaration";
import { EllipsisButton } from "#/components/features/conversation-panel/ellipsis-button";
import { BrandBadge } from "#/components/shared/badge";
import { cn } from "#/utils/utils";
import {
  settingsListIconActionButtonClassName,
  settingsListRowClassName,
} from "#/utils/settings-list-classes";

interface AgentProfileRowProps {
  profile: AgentProfileSummary;
  isActive: boolean;
  /** When false, the row is read-only and the actions menu is hidden. */
  canManage: boolean;
  onActivate: (profile: AgentProfileSummary) => void;
  onEdit: (profile: AgentProfileSummary) => void;
  onDelete: (profile: AgentProfileSummary) => void;
  isActivating: boolean;
}

export function AgentProfileRow({
  profile,
  isActive,
  canManage,
  onActivate,
  onEdit,
  onDelete,
  isActivating,
}: AgentProfileRowProps) {
  const { t } = useTranslation("openhands");
  const { backend } = useActiveBackend();
  // The server reseeds a deleted `default`, which would reset what it shares
  // with the global pages.
  const isProtected =
    backend.kind === "local" &&
    profile.name === WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME;
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
          className="min-w-0 max-w-full truncate text-sm font-medium text-contrast"
          title={profile.name}
        >
          {profile.name}
        </span>
        {secondary ? (
          <span
            className="min-w-0 max-w-full truncate text-sm text-muted"
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
            {t(I18nKey.SETTINGS$PROFILE_ACTIVE)}
          </BrandBadge>
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
              deleteDisabledReason={
                isProtected
                  ? t(I18nKey.SETTINGS$AGENT_PROFILE_DEFAULT_LOCKED)
                  : undefined
              }
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
