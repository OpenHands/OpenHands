import { Trans, useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import OpenHandsLogo from "#/assets/branding/openhands-logo.svg?react";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { BrandButton } from "../settings/brand-button";
import { useCreateBillingSession } from "#/hooks/mutation/use-create-billing-session";

export function SetupPaymentModal() {
  const { t } = useTranslation();
  const { mutate, isPending } = useCreateBillingSession();

  return (
    <ModalBackdrop>
      <ModalBody className="border border-tertiary">
        <OpenHandsLogo width={68} height={46} />
        <div className="flex flex-col gap-2 w-full items-center text-center">
          <h1 className="text-2xl font-bold">
            {t(I18nKey.BILLING$YOUVE_GOT_50)}
          </h1>
          <p>
            <Trans
              i18nKey="BILLING$CLAIM_YOUR_50"
              components={{ b: <strong /> }}
            />
          </p>
        </div>
        <BrandButton
          testId="proceed-to-stripe-button"
          type="submit"
          variant="primary"
          className="w-full"
          isDisabled={isPending}
          onClick={mutate}
        >
          {t(I18nKey.BILLING$PROCEED_TO_STRIPE)}
        </BrandButton>
      </ModalBody>
    </ModalBackdrop>
  );
}
