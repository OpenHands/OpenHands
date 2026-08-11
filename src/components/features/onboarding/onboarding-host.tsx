import { useLocation } from "react-router";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useLlmConfigured } from "#/hooks/use-llm-configured";
import { useOnboardingDismissal } from "./use-onboarding-dismissal";
import { useOnboardingVisibility } from "./use-onboarding-visibility";
import { OnboardingModal } from "./onboarding-modal";
import {
  isOnboardingPreviewActive,
  readOnboardingPreviewStep,
} from "./onboarding-preview";

/**
 * Mounts onboarding only when the active backend has no usable LLM. Local and
 * Cloud backends share the same readiness signal, so a configured shared
 * backend behaves consistently across browsers. Loading or indeterminate
 * readiness renders nothing to avoid a modal flash.
 *
 * Closing the flow records a backend-scoped dismissal for this browser
 * session; it does not persist a fake backend-readiness marker.
 *
 * With `?previewOnboardingStep=<0-3>` the modal opens on that slide for
 * design review without persisting completion (works on any route when
 * mounted from the root layout).
 */
export function OnboardingHost() {
  const location = useLocation();
  const previewStep = readOnboardingPreviewStep(location.search);
  const isPreview = isOnboardingPreviewActive(location.search);
  const { backend } = useActiveBackend();
  const { isConfigured, isLoading } = useLlmConfigured();
  const { isDismissed, markDismissed } = useOnboardingDismissal(backend.id);
  const shouldShow = useOnboardingVisibility({
    scopeKey: backend.id,
    eligible: !isLoading && !isConfigured,
    dismissed: isDismissed,
  });

  if (!isPreview && !shouldShow) return null;

  const handleClose = () => {
    if (isPreview) return;
    markDismissed();
  };

  return (
    <OnboardingModal
      onClose={handleClose}
      initialStep={previewStep ?? 0}
      isPreview={isPreview}
    />
  );
}
