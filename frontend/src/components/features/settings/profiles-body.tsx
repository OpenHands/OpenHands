import { useTranslation } from "react-i18next";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ProfileRow } from "#/components/features/settings/profile-row";
import { LlmProfileSummary } from "#/api/settings-service/profiles-service.api";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";

interface ProfilesBodyProps {
  isLoading: boolean;
  loadError: Error | null;
  profiles: LlmProfileSummary[];
  active: string | null;
  onActivate: (name: string) => void;
  onEdit: (profile: LlmProfileSummary) => void;
  onRename: (profile: LlmProfileSummary) => void;
  onDelete: (profile: LlmProfileSummary) => void;
  isActivating: boolean;
}

export function ProfilesBody({
  isLoading,
  loadError,
  profiles,
  active,
  onActivate,
  onEdit,
  onRename,
  onDelete,
  isActivating,
}: ProfilesBodyProps) {
  const { t } = useTranslation();

  if (isLoading) {
    return (
      <div className="flex justify-center p-4">
        <LoadingSpinner size="large" />
      </div>
    );
  }
  if (loadError) {
    return (
      <Typography.Paragraph className="text-sm text-red-400">
        {t(I18nKey.SETTINGS$PROFILES_LOAD_ERROR)}
      </Typography.Paragraph>
    );
  }
  if (profiles.length === 0) {
    return (
      <Typography.Paragraph className="text-sm text-gray-400 italic">
        {t(I18nKey.SETTINGS$PROFILES_EMPTY)}
      </Typography.Paragraph>
    );
  }
  return (
    <div className="border border-tertiary rounded-md divide-y divide-tertiary">
      {profiles.map((profile) => (
        <ProfileRow
          key={profile.name}
          profile={profile}
          isActive={profile.name === active}
          onActivate={onActivate}
          onEdit={onEdit}
          onRename={onRename}
          onDelete={onDelete}
          isActivating={isActivating}
        />
      ))}
    </div>
  );
}
