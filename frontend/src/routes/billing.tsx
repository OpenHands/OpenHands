import { redirect, useSearchParams } from "react-router";
import React from "react";
import { useTranslation } from "react-i18next";
import { PaymentForm } from "#/components/features/payment/payment-form";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";
import { useMe } from "#/hooks/query/use-me";
import { usePermission } from "#/hooks/organizations/use-permissions";
import { getActiveOrganizationUser } from "#/utils/org/permission-checks";
import { rolePermissions } from "#/utils/org/permissions";
import { isBillingHidden } from "#/utils/org/billing-visibility";
import { queryClient } from "#/query-client-config";
import OptionService from "#/api/option-service/option-service.api";
import { WebClientConfig } from "#/api/option-service/option.types";
import { QUERY_KEYS, CONFIG_CACHE_OPTIONS } from "#/hooks/query/query-keys";
import { getFirstAvailablePath } from "#/utils/settings-utils";

export const clientLoader = async () => {
  const config = await queryClient.fetchQuery<WebClientConfig>({
    queryKey: QUERY_KEYS.WEB_CLIENT_CONFIG,
    queryFn: OptionService.getConfig,
    ...CONFIG_CACHE_OPTIONS,
  });

  const isSaas = config?.app_mode === "saas";
  const featureFlags = config?.feature_flags;

  const getFallbackPath = () =>
    getFirstAvailablePath(isSaas, featureFlags) ?? "/settings";

  const user = await getActiveOrganizationUser();

  if (!user) {
    return redirect(getFallbackPath());
  }

  const userRole = user.role ?? "member";

  if (
    isBillingHidden(config, rolePermissions[userRole].includes("view_billing"))
  ) {
    return redirect(getFallbackPath());
  }

  return null;
};

function BillingSettingsScreen() {
  const { t } = useTranslation();
  const [searchParams, setSearchParams] = useSearchParams();
  const { data: me } = useMe();
  const { hasPermission } = usePermission(me?.role ?? "member");
  const canAddCredits = !!me && hasPermission("add_credits");
  const checkoutStatus = searchParams.get("checkout");
  const hasHandledCheckoutRef = React.useRef(false);

  React.useEffect(() => {
    if (!checkoutStatus) return;
    if (hasHandledCheckoutRef.current) return;
    hasHandledCheckoutRef.current = true;

    if (checkoutStatus === "success") {
      displaySuccessToast(t(I18nKey.PAYMENT$SUCCESS));
      setSearchParams({});
    } else if (checkoutStatus === "cancel") {
      displayErrorToast(t(I18nKey.PAYMENT$CANCELLED));
      setSearchParams({});
    }
  }, [checkoutStatus, setSearchParams, t]);

  return <PaymentForm isDisabled={!canAddCredits} />;
}

export default BillingSettingsScreen;
