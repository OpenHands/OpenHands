import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { BrandButton } from "#/components/features/settings/brand-button";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { I18nKey } from "#/i18n/declaration";
import { useUpdateOrganization } from "#/hooks/mutation/use-update-organization";
import { cn } from "#/utils/utils";
import { Typography } from "#/ui/typography";

interface ChangeOrgNameModalProps {
  onClose: () => void;
}

export function ChangeOrgNameModal({ onClose }: ChangeOrgNameModalProps) {
  const { t } = useTranslation();
  const { mutate: updateOrganization, isPending } = useUpdateOrganization();

  const formAction = (formData: FormData) => {
    const orgName = formData.get("org-name")?.toString();

    if (orgName?.trim()) {
      updateOrganization(orgName, {
        onSuccess: () => {
          onClose();
        },
      });
    }
  };

  return (
    <ModalBackdrop onClose={onClose}>
      <form
        action={formAction}
        data-testid="update-org-name-form"
        className="bg-base rounded-xl p-4 border w-sm border-tertiary items-start flex flex-col gap-6"
      >
        <div className="flex flex-col gap-2 w-full">
          <Typography.H3 className="text-lg text-white font-semibold">
            {t(I18nKey.ORG$CHANGE_ORG_NAME)}
          </Typography.H3>
          <Typography.Text className="text-xs text-gray-400">
            {t(I18nKey.ORG$MODIFY_ORG_NAME_DESCRIPTION)}
          </Typography.Text>
          <SettingsInput
            name="org-name"
            type="text"
            required
            isDisabled={isPending}
            className="w-full"
            label={t(I18nKey.ORG$ORGANIZATION_NAME)}
            placeholder={t(I18nKey.ORG$ENTER_NEW_ORGANIZATION_NAME)}
          />
        </div>

        <div className="flex items-center gap-2 w-full">
          <BrandButton
            variant="primary"
            type="submit"
            isDisabled={isPending}
            className={cn(
              "flex-1",
              isPending && "flex text-white justify-center",
            )}
          >
            {isPending ? (
              <LoadingSpinner size="small" />
            ) : (
              t(I18nKey.BUTTON$SAVE)
            )}
          </BrandButton>
        </div>
      </form>
    </ModalBackdrop>
  );
}
