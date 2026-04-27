import React from "react";
import { useLocation, useNavigate } from "react-router";
import { useOnboardingStatus } from "#/hooks/query/use-onboarding-status";

/**
 * Forces SaaS users with incomplete onboarding to /onboarding before they can
 * access any protected route. Mirrors EmailVerificationGuard.
 */
export function OnboardingGuard({ children }: { children: React.ReactNode }) {
  const { data, isLoading } = useOnboardingStatus();
  const navigate = useNavigate();
  const { pathname } = useLocation();

  React.useEffect(() => {
    if (isLoading) return;
    if (data?.should_complete_onboarding && pathname !== "/onboarding") {
      navigate("/onboarding", { replace: true });
    }
  }, [data?.should_complete_onboarding, isLoading, pathname, navigate]);

  return children;
}
