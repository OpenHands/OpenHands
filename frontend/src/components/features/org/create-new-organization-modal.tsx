import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { useCreateOrganization } from "#/hooks/mutation/use-create-organization";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { I18nKey } from "#/i18n/declaration";

interface CreateNewOrganizationModalProps {
  onClose: () => void;
  onSuccess?: () => void;
}

export function CreateNewOrganizationModal({
  onClose,
  onSuccess,
}: CreateNewOrganizationModalProps) {
  const { t } = useTranslation();
  const { mutate: createOrganization } = useCreateOrganization();
  const { setOrgId } = useSelectedOrganizationId();

  const formAction = (formData: FormData) => {
    const orgName = formData.get("org-name")?.toString();
    if (orgName) {
      createOrganization(
        { name: orgName },
        {
          onSuccess: (newOrg) => {
            setOrgId(newOrg.id);
            onClose();
            onSuccess?.();
          },
        },
      );
    }
  };

  return (
    <ModalBackdrop onClose={onClose}>
      <div
        data-testid="create-org-modal"
        className="bg-base rounded-xl p-4 border w-sm border-tertiary items-start"
        onClick={(e) => e.stopPropagation()}
      >
        <form action={formAction}>
          <label>
            {t(I18nKey.ORG$ORGANIZATION_NAME)}
            <input data-testid="org-name-input" name="org-name" type="text" />
          </label>

          <button type="submit">{t(I18nKey.ORG$NEXT)}</button>
          <button type="button" onClick={onClose}>
            {t(I18nKey.ORG$SKIP)}
          </button>
        </form>
      </div>
    </ModalBackdrop>
  );
}
