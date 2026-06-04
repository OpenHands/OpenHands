import React from "react";
import { useTranslation } from "react-i18next";
import { useCreateStripeCheckoutSession } from "#/hooks/mutation/stripe/use-create-stripe-checkout-session";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalButtonGroup } from "#/components/shared/modals/modal-button-group";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { I18nKey } from "#/i18n/declaration";
import {
  amountIsValid,
  getAmountValidationError,
  MAXIMUM_AMOUNT,
  MINIMUM_AMOUNT,
} from "#/utils/amount-is-valid";

interface AddCreditsModalProps {
  onClose: () => void;
}

export function AddCreditsModal({ onClose }: AddCreditsModalProps) {
  const { t } = useTranslation();
  const { mutate: addBalance } = useCreateStripeCheckoutSession();

  const [inputValue, setInputValue] = React.useState("");
  const [errorMessage, setErrorMessage] = React.useState<string | null>(null);

  const getErrorMessage = (value: string): string | null => {
    const error = getAmountValidationError(value);

    switch (error) {
      case null:
      case "empty":
        return null;
      case "invalid":
        return t(I18nKey.PAYMENT$ERROR_INVALID_NUMBER);
      case "negative":
        return t(I18nKey.PAYMENT$ERROR_NEGATIVE_AMOUNT);
      case "not_integer":
        return t(I18nKey.PAYMENT$ERROR_MUST_BE_WHOLE_NUMBER);
      case "below_minimum":
        return t(I18nKey.PAYMENT$ERROR_MINIMUM_AMOUNT);
      case "above_maximum":
        return t(I18nKey.PAYMENT$ERROR_MAXIMUM_AMOUNT);
    }
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
            min={MINIMUM_AMOUNT}
            max={MAXIMUM_AMOUNT}
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
