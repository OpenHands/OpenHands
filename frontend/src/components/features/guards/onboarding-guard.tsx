import React from "react";
import { useLocation, useNavigate } from "react-router";
import { useOnboardingStatus } from "#/hooks/query/use-onboarding-status";
import { useConfig } from "#/hooks/query/use-config";

/**
 * Forces SaaS users with incomplete onboarding to /onboarding before they can
 * access any protected route. Mirrors EmailVerificationGuard.
 */
export function OnboardingGuard({ children }: { children: React.ReactNode }) {
  const { data, isLoading } = useOnboardingStatus();
  const { data: config } = useConfig();
  const navigate = useNavigate();
  const { pathname } = useLocation();

  React.useEffect(() => {
    if (isLoading) return;
    // Only redirect to onboarding if the feature flag is enabled
    if (
      config?.feature_flags?.enable_onboarding &&
      data?.should_complete_onboarding &&
      pathname !== "/onboarding"
    ) {
      navigate("/onboarding", { replace: true });
    }
  }, [
    config?.feature_flags?.enable_onboarding,
    data?.should_complete_onboarding,
    isLoading,
    pathname,
    navigate,
  ]);

  return children;
}
