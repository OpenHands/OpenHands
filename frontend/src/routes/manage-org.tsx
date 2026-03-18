import React from "react";
import { useTranslation } from "react-i18next";
import { useCreateStripeCheckoutSession } from "#/hooks/mutation/stripe/use-create-stripe-checkout-session";
import { useOrganization } from "#/hooks/query/use-organization";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalButtonGroup } from "#/components/shared/modals/modal-button-group";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { useMe } from "#/hooks/query/use-me";
import { useConfig } from "#/hooks/query/use-config";
import { I18nKey } from "#/i18n/declaration";
import { amountIsValid } from "#/utils/amount-is-valid";
import { CreditsChip } from "#/ui/credits-chip";
import { InteractiveChip } from "#/ui/interactive-chip";
import { usePermission } from "#/hooks/organizations/use-permissions";
import { createPermissionGuard } from "#/utils/org/permission-guard";
import { isBillingHidden } from "#/utils/org/billing-visibility";
import { DeleteOrgConfirmationModal } from "#/components/features/org/delete-org-confirmation-modal";
import { ChangeOrgNameModal } from "#/components/features/org/change-org-name-modal";
import { useBalance } from "#/hooks/query/use-balance";
import { cn } from "#/utils/utils";

interface AddCreditsModalProps {
  onClose: () => void;
}

function AddCreditsModal({ onClose }: AddCreditsModalProps) {
  const { t } = useTranslation();
  const { mutate: addBalance } = useCreateStripeCheckoutSession();

  const [inputValue, setInputValue] = React.useState("");
  const [errorMessage, setErrorMessage] = React.useState<string | null>(null);

  const getErrorMessage = (value: string): string | null => {
    if (!value.trim()) return null;

    const numValue = parseInt(value, 10);
    if (Number.isNaN(numValue)) {
      return t(I18nKey.PAYMENT$ERROR_INVALID_NUMBER);
    }
    if (numValue < 0) {
      return t(I18nKey.PAYMENT$ERROR_NEGATIVE_AMOUNT);
    }
    if (numValue < 10) {
      return t(I18nKey.PAYMENT$ERROR_MINIMUM_AMOUNT);
    }
    if (numValue > 25000) {
      return t(I18nKey.PAYMENT$ERROR_MAXIMUM_AMOUNT);
    }
    if (numValue !== parseFloat(value)) {
      return t(I18nKey.PAYMENT$ERROR_MUST_BE_WHOLE_NUMBER);
    }
    return null;
  };

  const formAction = (formData: FormData) => {
    const amount = formData.get("amount")?.toString();

    if (amount?.trim()) {
      if (!amountIsValid(amount)) {
        const error = getErrorMessage(amount);
        setErrorMessage(error || "Invalid amount");
        return;
      }

      const intValue = parseInt(amount, 10);

      addBalance({ amount: intValue }, { onSuccess: onClose });

      setErrorMessage(null);
    }
  };

  const handleAmountInputChange = (value: string) => {
    setInputValue(value);
    setErrorMessage(null);
  };

  return (
    <ModalBackdrop onClose={onClose}>
      <form
        data-testid="add-credits-form"
        action={formAction}
        noValidate
        className="w-sm rounded-xl bg-base-secondary flex flex-col p-6 gap-4 border border-tertiary"
      >
        <h3 className="text-xl font-bold">{t(I18nKey.ORG$ADD_CREDITS)}</h3>
        <div className="flex flex-col gap-2">
          <SettingsInput
            testId="amount-input"
            name="amount"
            label={t(I18nKey.PAYMENT$SPECIFY_AMOUNT_USD)}
            type="number"
            min={10}
            max={25000}
            step={1}
            value={inputValue}
            onChange={(value) => handleAmountInputChange(value)}
            className="w-full"
          />
          {errorMessage && (
            <p className="text-red-500 text-sm mt-1" data-testid="amount-error">
              {errorMessage}
            </p>
          )}
        </div>

        <ModalButtonGroup
          primaryText={t(I18nKey.ORG$NEXT)}
          onSecondaryClick={onClose}
          primaryType="submit"
        />
      </form>
    </ModalBackdrop>
  );
}

export const clientLoader = createPermissionGuard("view_billing");

function ManageOrg() {
  const { t } = useTranslation();
  const { data: me } = useMe();
  const { data: organization } = useOrganization();
  const { data: balance } = useBalance();
  const { data: config } = useConfig();

  const role = me?.role ?? "member";
  const { hasPermission } = usePermission(role);

  const [addCreditsFormVisible, setAddCreditsFormVisible] =
    React.useState(false);
  const [changeOrgNameFormVisible, setChangeOrgNameFormVisible] =
    React.useState(false);
  const [deleteOrgConfirmationVisible, setDeleteOrgConfirmationVisible] =
    React.useState(false);

  const canChangeOrgName = !!me && hasPermission("change_organization_name");
  const canDeleteOrg = !!me && hasPermission("delete_organization");
  const canAddCredits = !!me && hasPermission("add_credits");
  const shouldHideBilling = isBillingHidden(
    config,
    hasPermission("view_billing"),
  );

  return (
    <div
      data-testid="manage-org-screen"
      className="flex flex-col items-start gap-6"
    >
      {changeOrgNameFormVisible && (
        <ChangeOrgNameModal
          onClose={() => setChangeOrgNameFormVisible(false)}
        />
      )}
      {deleteOrgConfirmationVisible && (
        <DeleteOrgConfirmationModal
          onClose={() => setDeleteOrgConfirmationVisible(false)}
        />
      )}

      {!shouldHideBilling && (
        <div className="flex flex-col gap-2">
          <span className="text-white text-xs font-semibold">
            {t(I18nKey.ORG$CREDITS)}
          </span>
          <div className="flex items-center gap-2">
            <CreditsChip testId="available-credits">
              ${Number(balance ?? 0).toFixed(2)}
            </CreditsChip>
            {canAddCredits && (
              <InteractiveChip onClick={() => setAddCreditsFormVisible(true)}>
                {t(I18nKey.ORG$ADD)}
              </InteractiveChip>
            )}
          </div>
        </div>
      )}

      {addCreditsFormVisible && !shouldHideBilling && (
        <AddCreditsModal onClose={() => setAddCreditsFormVisible(false)} />
      )}

      <div data-testid="org-name" className="flex flex-col gap-2 w-sm">
        <span className="text-white text-xs font-semibold">
          {t(I18nKey.ORG$ORGANIZATION_NAME)}
        </span>

        <div
          className={cn(
            "text-sm p-3 bg-modal-input rounded",
            "flex items-center justify-between",
          )}
        >
          <span className="text-white">{organization?.name}</span>
          {canChangeOrgName && (
            <button
              type="button"
              onClick={() => setChangeOrgNameFormVisible(true)}
              className="text-sm text-org-text font-normal leading-5 hover:text-white transition-colors cursor-pointer"
            >
              {t(I18nKey.ORG$CHANGE)}
            </button>
          )}
        </div>
      </div>

      {canDeleteOrg && (
        <button
          type="button"
          onClick={() => setDeleteOrgConfirmationVisible(true)}
          className="text-xs text-[#FF3B30] cursor-pointer font-semibold hover:underline"
        >
          {t(I18nKey.ORG$DELETE_ORGANIZATION)}
        </button>
      )}
    </div>
  );
}

export default ManageOrg;
