import { useQuery } from "@tanstack/react-query";
import { useConfig } from "./use-config";
import BillingService from "#/api/billing-service/billing-service.api";
import { useIsOnTosPage } from "#/hooks/use-is-on-tos-page";

export const useSubscriptionAccess = () => {
  const { data: config } = useConfig();
  const isOnTosPage = useIsOnTosPage();

  return useQuery({
    queryKey: ["user", "subscription_access"],
    queryFn: BillingService.getSubscriptionAccess,
    enabled:
      !isOnTosPage &&
      config?.app_mode === "saas" &&
      config?.feature_flags?.enable_billing,
  });
};
