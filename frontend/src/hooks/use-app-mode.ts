import { useMemo } from "react";
import { useConfig } from "#/hooks/query/use-config";

/**
 * Hook that provides boolean checks for app mode deployment mode.
 *
 * App Mode (app_mode):
 * - "oss": Open source version running locally/self-hosted
 * - "saas": All-Hands managed SaaS version
 *
 * Deployment Mode (deployment_mode):
 * - "cloud": Enterprise customers running on All-Hands managed infrastructure (*.all-hands.dev, *.openhands.ai)
 * - "self_hosted": Enterprise customers running on their own infrastructure
 *
 * Note: SaaS mode can have either cloud or self_hosted deployment mode.
 */
export function useAppMode() {
  const { data: config } = useConfig();

  return useMemo(() => {
    const appMode = config?.app_mode;
    const deploymentMode = config?.feature_flags?.deployment_mode;

    return {
      // App Mode checks
      isOss: appMode === "oss",
      isSaas: appMode === "saas",

      // Deployment Mode checks
      isCloud: deploymentMode === "cloud",
      isSelfHosted: deploymentMode === "self_hosted",

      /** Enterprise checks */
      isEnterpriseSelfHosted:
        appMode === "saas" && deploymentMode === "self_hosted",
      isEnterpriseCloud: appMode === "saas" && deploymentMode === "cloud",

      // Raw values (for cases where the actual value is needed)
      appMode,
      deploymentMode,
    };
  }, [config?.app_mode, config?.feature_flags?.deployment_mode]);
}
