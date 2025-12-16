import React from "react";
import { redirect } from "react-router";
import { useTranslation } from "react-i18next";
import { useCreateStripeCheckoutSession } from "#/hooks/mutation/stripe/use-create-stripe-checkout-session";
import { useOrganization } from "#/hooks/query/use-organization";
import { useOrganizationPaymentInfo } from "#/hooks/query/use-organization-payment-info";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { cn } from "#/utils/utils";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { BrandButton } from "#/components/features/settings/brand-button";
import { useMe } from "#/hooks/query/use-me";
import { rolePermissions } from "#/utils/org/permissions";
import {
  getSelectedOrgFromQueryClient,
  getMeFromQueryClient,
} from "#/utils/query-client-getters";
import { queryClient } from "#/query-client-config";
import { I18nKey } from "#/i18n/declaration";
import { amountIsValid } from "#/utils/amount-is-valid";
import { useUpdateOrganization } from "#/hooks/mutation/use-update-organization";
import { useDeleteOrganization } from "#/hooks/mutation/use-delete-organization";

function TempChip({
  children,
  ...props
}: React.PropsWithChildren<{ "data-testid": string }>) {
  return (
    <div
      // eslint-disable-next-line react/jsx-props-no-spreading
      {...props}
      style={{ minWidth: "100px" }}
      data-openhands-chip
      className="bg-[#FFE165] px-4 rounded-[100px] text-black text-lg text-center font-semibold"
    >
      {children}
    </div>
  );
}

interface TempInteractiveChipProps {
  onClick: () => void;
}

function TempInteractiveChip({
  children,
  onClick,
}: React.PropsWithChildren<TempInteractiveChipProps>) {
  return (
    <div
      onClick={onClick}
      className="bg-[#E4E4E4] px-2 rounded-[100px] text-black text-sm text-center font-semibold cursor-pointer"
    >
      {children}
    </div>
  );
}

function TempButton({
  children,
  onClick,
  type,
  variant = "primary",
}: React.PropsWithChildren<{
  onClick?: () => void;
  type: "button" | "submit";
  variant?: "primary" | "secondary";
}>) {
  return (
    <button
      className={cn(
        "flex-1 py-3 rounded text-sm text-center font-semibold cursor-pointer",
        variant === "primary" && "bg-[#F3CE49] text-black",
        variant === "secondary" && "bg-[#737373] text-white",
      )}
      type={type === "submit" ? "submit" : "button"}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

interface ChangeOrgNameModalProps {
  onClose: () => void;
}

function ChangeOrgNameModal({ onClose }: ChangeOrgNameModalProps) {
  const { t } = useTranslation();
  const { mutate: updateOrganization } = useUpdateOrganization();

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
        className={cn(
          "bg-base rounded-xl p-4 border w-sm border-tertiary items-start",
          "flex flex-col gap-6",
        )}
      >
        <div className="flex flex-col gap-2 w-full">
          <h3 className="text-lg font-semibold">
            {t(I18nKey.ORG$CHANGE_ORG_NAME)}
          </h3>
          <p className="text-xs text-gray-400">
            {t(I18nKey.ORG$MODIFY_ORG_NAME_DESCRIPTION)}
          </p>
          <SettingsInput
            name="org-name"
            type="text"
            required
            className="w-full"
            label="Organization Name"
            placeholder="Enter new organization name"
          />
        </div>

        <BrandButton variant="primary" type="submit" className="w-full">
          {t(I18nKey.BUTTON$SAVE)}
        </BrandButton>
      </form>
    </ModalBackdrop>
  );
}

interface DeleteOrgConfirmationModalProps {
  onClose: () => void;
}

function DeleteOrgConfirmationModal({
  onClose,
}: DeleteOrgConfirmationModalProps) {
  const { t } = useTranslation();
  const { mutate: deleteOrganization } = useDeleteOrganization();

  return (
    <div data-testid="delete-org-confirmation">
      <button
        type="button"
        onClick={() =>
          deleteOrganization(undefined, {
            onSuccess: onClose,
          })
        }
      >
        {t(I18nKey.BUTTON$CONFIRM)}
      </button>
    </div>
  );
}

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
    // Clear error message when user starts typing again
    setErrorMessage(null);
  };

  return (
    <ModalBackdrop>
      <form
        data-testid="add-credits-form"
        action={formAction}
        noValidate
        className="w-md rounded-xl bg-[#171717] flex flex-col p-6 gap-6"
      >
        <div className="flex flex-col gap-2">
          <h3 className="text-xl font-semibold">
            {t(I18nKey.ORG$ADD_CREDITS)}
          </h3>
          <input
            data-testid="amount-input"
            name="amount"
            type="number"
            className="text-lg bg-[#27272A] p-2"
            placeholder={t(I18nKey.PAYMENT$SPECIFY_AMOUNT_USD)}
            min={10}
            max={25000}
            step={1}
            value={inputValue}
            onChange={(e) => handleAmountInputChange(e.target.value)}
          />
          {errorMessage && (
            <p className="text-red-500 text-sm mt-1" data-testid="amount-error">
              {errorMessage}
            </p>
          )}
        </div>

        <div className="flex gap-2">
          <TempButton type="submit">{t(I18nKey.ORG$NEXT)}</TempButton>
          <TempButton type="button" onClick={onClose} variant="secondary">
            {t(I18nKey.BUTTON$CANCEL)}
          </TempButton>
        </div>
      </form>
    </ModalBackdrop>
  );
}

export const clientLoader = async () => {
  const selectedOrgId = getSelectedOrgFromQueryClient();
  let me = getMeFromQueryClient(selectedOrgId);

  if (!me && selectedOrgId) {
    me = await organizationService.getMe({ orgId: selectedOrgId });
    queryClient.setQueryData(["organizations", selectedOrgId, "me"], me);
  }

  if (!me || me.role === "user") {
    // if user is USER role, redirect to user settings
    return redirect("/settings/user");
  }

  return null;
};

function ManageOrg() {
  const { t } = useTranslation();
  const { data: me } = useMe();
  const { data: organization } = useOrganization();
  const { data: organizationPaymentInfo } = useOrganizationPaymentInfo();

  const [addCreditsFormVisible, setAddCreditsFormVisible] =
    React.useState(false);
  const [changeOrgNameFormVisible, setChangeOrgNameFormVisible] =
    React.useState(false);
  const [deleteOrgConfirmationVisible, setDeleteOrgConfirmationVisible] =
    React.useState(false);

  const canChangeOrgName =
    !!me && rolePermissions[me.role].includes("change_organization_name");
  const canDeleteOrg =
    !!me && rolePermissions[me.role].includes("delete_organization");
  const canAddCredits =
    !!me && rolePermissions[me.role].includes("add_credits");

  return (
    <div
      data-testid="manage-org-screen"
      className="flex flex-col items-start gap-6 px-11 py-6"
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

      <div className="flex flex-col gap-2">
        <span className="text-white text-xs font-semibold ml-1">
          {t(I18nKey.ORG$CREDITS)}
        </span>
        <div className="flex items-center gap-2">
          <TempChip data-testid="available-credits">
            {organization?.credits}
          </TempChip>
          {canAddCredits && (
            <TempInteractiveChip onClick={() => setAddCreditsFormVisible(true)}>
              {t(I18nKey.ORG$ADD)}
            </TempInteractiveChip>
          )}
        </div>
      </div>

      {addCreditsFormVisible && (
        <AddCreditsModal onClose={() => setAddCreditsFormVisible(false)} />
      )}

      <div data-testid="org-name" className="flex flex-col gap-2 w-sm">
        <span className="text-white text-xs font-semibold ml-1">
          {t(I18nKey.ORG$ORGANIZATION_NAME)}
        </span>

        <div
          className={cn(
            "text-sm p-3 bg-base rounded",
            "flex items-center justify-between",
          )}
        >
          <span className="text-white">{organization?.name}</span>
          {canChangeOrgName && (
            <button
              type="button"
              onClick={() => setChangeOrgNameFormVisible(true)}
              className="text-[#A3A3A3] hover:text-white transition-colors cursor-pointer"
            >
              {t(I18nKey.ORG$CHANGE)}
            </button>
          )}
        </div>
      </div>

      <div className="flex flex-col gap-2 w-sm">
        <span className="text-white text-xs font-semibold ml-1">
          {t(I18nKey.ORG$BILLING_INFORMATION)}
        </span>

        <span
          data-testid="billing-info"
          className={cn(
            "text-sm p-3 bg-base rounded text-[#A3A3A3]",
            "flex items-center justify-between",
          )}
        >
          {organizationPaymentInfo?.cardNumber}
        </span>
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
