import { useState } from "react";
import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
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
  const [orgName, setOrgName] = useState<string>("");

  const formAction = () => {
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
        className="rounded-xl p-6 w-sm items-start flex flex-col gap-6 bg-[#171717] modal-box-shadow"
      >
        <div className="flex flex-col gap-2 w-full">
          <Typography.H3 className="text-xl leading-6 font-semibold">
            {t(I18nKey.ORG$CHANGE_ORG_NAME)}
          </Typography.H3>
          <Typography.Text className="text-xs leading-4 font-normal text-[#A3A3A3]">
            {t(I18nKey.ORG$MODIFY_ORG_NAME_DESCRIPTION)}
          </Typography.Text>
          <div className="rounded w-full p-2 placeholder:text-tertiary-alt bg-[#27272A] border-none pl-3">
            <input
              data-testid="org-name"
              value={orgName}
              placeholder={t(I18nKey.ORG$ENTER_NEW_ORGANIZATION_NAME)}
              onChange={(e) => setOrgName(e.target.value)}
              className="w-full text-sm leading-4 font-normal outline-none bg-transparent"
            />
          </div>
        </div>

        <div className="flex items-center gap-2 w-full">
          <BrandButton
            variant="primary"
            type="submit"
            isDisabled={isPending}
            className={cn(
              "flex-1 flex items-center justify-center bg-[#F3CE49] text-sm leading-4 font-medium rounded h-10",
              isPending && "flex text-white justify-center",
            )}
          >
            {isPending ? (
              <LoadingSpinner
                size="small"
                className="w-5 h-5"
                innerClassName="hidden"
                outerClassName="w-5 h-5"
              />
            ) : (
              t(I18nKey.BUTTON$SAVE)
            )}
          </BrandButton>
          <BrandButton
            type="button"
            variant="secondary"
            onClick={onClose}
            className="flex-1 bg-[#737373] text-sm text-white leading-4 font-medium rounded border-none h-10"
            isDisabled={isPending}
          >
            {t(I18nKey.BUTTON$CLOSE)}
          </BrandButton>
        </div>
      </form>
    </ModalBackdrop>
  );
}
