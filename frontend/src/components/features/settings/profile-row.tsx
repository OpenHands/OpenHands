import { useState } from "react";
import { useTranslation } from "react-i18next";
import { ProfileActionsMenu } from "#/components/features/settings/profile-actions-menu";
import { LlmProfileSummary } from "#/api/settings-service/profiles-service.api";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import ThreeDotsVerticalIcon from "#/icons/three-dots-vertical.svg?react";

interface ProfileRowProps {
  profile: LlmProfileSummary;
  isActive: boolean;
  onActivate: (name: string) => void;
  onEdit: (profile: LlmProfileSummary) => void;
  onRename: (profile: LlmProfileSummary) => void;
  onDelete: (profile: LlmProfileSummary) => void;
  isActivating: boolean;
}

export function ProfileRow({
  profile,
  isActive,
  onActivate,
  onEdit,
  onRename,
  onDelete,
  isActivating,
}: ProfileRowProps) {
  const { t } = useTranslation();
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div
      data-testid="profile-row"
      className="flex items-center justify-between gap-3 px-5 py-4"
    >
      <div className="flex flex-col gap-1 min-w-0 flex-1 sm:flex-row sm:items-center sm:gap-3">
        <Typography.Text
          className="font-medium text-white truncate min-w-0 max-w-full"
          title={profile.name}
        >
          {profile.name}
        </Typography.Text>
        {profile.model ? (
          <Typography.Text
            className="text-sm text-gray-400 truncate min-w-0 max-w-full"
            title={profile.model}
          >
            {profile.model}
          </Typography.Text>
        ) : null}
        {isActive && (
          <Typography.Text
            className="text-xs bg-primary text-[#0D0F11] font-semibold rounded-full px-2 py-0.5 whitespace-nowrap self-start sm:self-auto"
            testId="profile-active-badge"
          >
            {t(I18nKey.SETTINGS$PROFILE_ACTIVE_BADGE)}
          </Typography.Text>
        )}
      </div>
      <div className="relative shrink-0">
        <button
          type="button"
          onClick={() => setMenuOpen((open) => !open)}
          aria-label={t(I18nKey.SETTINGS$PROFILE_MENU)}
          className="cursor-pointer text-gray-300 hover:text-white p-2 border border-tertiary rounded-md"
          data-testid="profile-menu-trigger"
        >
          <ThreeDotsVerticalIcon width={16} height={16} />
        </button>
        {menuOpen && (
          <ProfileActionsMenu
            onEdit={() => onEdit(profile)}
            onRename={() => onRename(profile)}
            onSetActive={() => onActivate(profile.name)}
            onDelete={() => onDelete(profile)}
            isActive={isActive}
            isActivating={isActivating}
            onClose={() => setMenuOpen(false)}
          />
        )}
      </div>
    </div>
  );
}
