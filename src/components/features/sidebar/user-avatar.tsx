import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import ProfileIcon from "#/icons/profile.svg?react";
import { cn } from "#/utils/utils";
import { Avatar } from "./avatar";

interface UserAvatarProps {
  avatarUrl?: string;
  isLoading?: boolean;
}

export function UserAvatar({ avatarUrl, isLoading }: UserAvatarProps) {
  const { t } = useTranslation("openhands");

  return (
    <button
      type="button"
      data-testid="user-avatar"
      className={cn(
        "w-8 h-8 rounded-full flex items-center justify-center cursor-pointer",
        isLoading && "bg-transparent",
      )}
    >
      {!isLoading && avatarUrl && <Avatar src={avatarUrl} />}
      {!isLoading && !avatarUrl && (
        <ProfileIcon
          aria-label={t(I18nKey.USER$AVATAR_PLACEHOLDER)}
          width={28}
          height={28}
          className="text-[var(--oh-muted)]"
        />
      )}
      {isLoading && <LoadingSpinner size="small" />}
    </button>
  );
}
