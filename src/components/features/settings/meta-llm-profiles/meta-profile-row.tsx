import { useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { MetaProfileActionsMenu } from "./meta-profile-actions-menu";
import { MetaProfileInfo } from "#/api/meta-profiles-service/meta-profiles-service.api";
import { I18nKey } from "#/i18n/declaration";
import { EllipsisButton } from "#/components/features/conversation-panel/ellipsis-button";
import { BrandBadge } from "#/components/shared/badge";
import { cn } from "#/utils/utils";
import {
  settingsListIconActionButtonClassName,
  settingsListRowClassName,
} from "#/utils/settings-list-classes";

interface MetaProfileRowProps {
  info: MetaProfileInfo;
  isActive: boolean;
  onActivate: (name: string) => void;
  onEdit: (name: string) => void;
  onDelete: (name: string) => void;
  isActivating: boolean;
}

export function MetaProfileRow({
  info,
  isActive,
  onActivate,
  onEdit,
  onDelete,
  isActivating,
}: MetaProfileRowProps) {
  const { t } = useTranslation("openhands");
  const [menuOpen, setMenuOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);

  const route = [info.classifier_model, info.default_model]
    .filter(Boolean)
    .join(" → ");
  const summary = `${route}${route ? " · " : ""}${info.num_classes} ${t(
    I18nKey.SETTINGS$META_PROFILE_CLASSES,
  )}`;

  return (
    <div
      data-testid={`meta-profile-row-${info.name}`}
      className={cn(settingsListRowClassName, "justify-between gap-3")}
    >
      <div className="flex min-w-0 flex-1 items-center gap-3">
        <span
          className="min-w-0 max-w-full truncate text-sm font-medium text-white"
          title={info.name}
        >
          {info.name}
        </span>
        <span
          className="min-w-0 max-w-full truncate text-sm text-[var(--oh-muted)]"
          title={summary}
        >
          {summary}
        </span>
        {isActive && (
          <BrandBadge
            className="shrink-0 whitespace-nowrap px-2.5 py-1 text-xs"
            data-testid="meta-profile-active-badge"
          >
            {t(I18nKey.SETTINGS$META_PROFILE_ACTIVE)}
          </BrandBadge>
        )}
      </div>
      <div className="relative shrink-0">
        <EllipsisButton
          ref={triggerRef}
          onClick={() => setMenuOpen((open) => !open)}
          ariaLabel={t(I18nKey.SETTINGS$PROFILE_MENU)}
          testId={`meta-profile-menu-trigger-${info.name}`}
          className={settingsListIconActionButtonClassName}
        />
        {menuOpen && (
          <MetaProfileActionsMenu
            anchorRef={triggerRef}
            onEdit={() => onEdit(info.name)}
            onSetActive={() => onActivate(info.name)}
            onDelete={() => onDelete(info.name)}
            isActive={isActive}
            isActivating={isActivating}
            onClose={() => setMenuOpen(false)}
          />
        )}
      </div>
    </div>
  );
}
