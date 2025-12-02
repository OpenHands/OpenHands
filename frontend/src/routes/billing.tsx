import { useSearchParams } from "react-router";
import React from "react";
import { useTranslation } from "react-i18next";
import { PaymentForm } from "#/components/features/payment/payment-form";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";
import { useTracking } from "#/hooks/use-tracking";

function BillingSettingsScreen() {
  const { t } = useTranslation();
  const [searchParams, setSearchParams] = useSearchParams();
  const { trackCreditsPurchased } = useTracking();
  const checkoutStatus = searchParams.get("checkout");

  React.useEffect(() => {
    if (checkoutStatus === "success") {
      // Get purchase details from URL params
      const amount = searchParams.get("amount");
      const sessionId = searchParams.get("session_id");

      // Track credits purchased if we have the necessary data
      if (amount && sessionId) {
        trackCreditsPurchased({
          amountUsd: parseFloat(amount),
          stripeSessionId: sessionId,
        });
      }

      displaySuccessToast(t(I18nKey.PAYMENT$SUCCESS));

      setSearchParams({});
    } else if (checkoutStatus === "cancel") {
      displayErrorToast(t(I18nKey.PAYMENT$CANCELLED));
      setSearchParams({});
    }
  }, [checkoutStatus, searchParams, setSearchParams, t, trackCreditsPurchased]);

  return <PaymentForm />;
}

export default BillingSettingsScreen;
