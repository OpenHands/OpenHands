import React from "react";
import { useTranslation } from "react-i18next";
import { useCreateStripeCheckoutSession } from "#/hooks/mutation/stripe/use-create-stripe-checkout-session";
import { useBalance } from "#/hooks/query/use-balance";
import { cn } from "#/utils/utils";
import MoneyIcon from "#/icons/money.svg?react";
import { SettingsInput } from "../settings/settings-input";
import { BrandButton } from "../settings/brand-button";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { amountIsValid } from "#/utils/amount-is-valid";
import { I18nKey } from "#/i18n/declaration";
import { PoweredByStripeTag } from "./powered-by-stripe-tag";

export function PaymentForm() {
  const { t } = useTranslation();
  const { data: balance, isLoading } = useBalance();
  const { mutate: addBalance, isPending } = useCreateStripeCheckoutSession();

  const [buttonIsDisabled, setButtonIsDisabled] = React.useState(true);

  const billingFormAction = async (formData: FormData) => {
    const amount = formData.get("top-up-input")?.toString();

    if (amount?.trim()) {
      if (!amountIsValid(amount)) return;

      const intValue = parseInt(amount, 10);
      addBalance({ amount: intValue });
    }

    setButtonIsDisabled(true);
  };

  const handleTopUpInputChange = (value: string) => {
    setButtonIsDisabled(!amountIsValid(value));
  };

  return (
    <form
      action={billingFormAction}
      data-testid="billing-settings"
      className="flex flex-col gap-6"
    >
      <div
        className={cn(
          "flex items-center justify-between w-[680px] bg-[#7F7445] rounded-sm px-3 py-2",
          "text-[28px] leading-8 -tracking-[0.02em] font-bold",
        )}
      >
        <div className="flex items-center gap-2">
          <MoneyIcon width={22} height={14} />
          <span>{t(I18nKey.PAYMENT$MANAGE_CREDITS)}</span>
        </div>
        {!isLoading && (
          <span data-testid="user-balance">${Number(balance).toFixed(2)}</span>
        )}
        {isLoading && <LoadingSpinner size="small" />}
      </div>

      <div className="flex flex-col gap-3">
        <SettingsInput
          testId="top-up-input"
          name="top-up-input"
          onChange={handleTopUpInputChange}
          type="number"
          label={t(I18nKey.PAYMENT$ADD_FUNDS)}
          placeholder={t(I18nKey.PAYMENT$SPECIFY_AMOUNT_USD)}
          className="w-[680px]"
          min={10}
          max={25000}
          step={1}
        />

        <div className="flex items-center w-[680px] gap-2">
          <BrandButton
            variant="primary"
            type="submit"
            isDisabled={isPending || buttonIsDisabled}
          >
            {t(I18nKey.PAYMENT$ADD_CREDIT)}
          </BrandButton>
          {isPending && <LoadingSpinner size="small" />}
          <PoweredByStripeTag />
        </div>
      </div>
    </form>
  );
}
