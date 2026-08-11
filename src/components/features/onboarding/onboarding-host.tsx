import { useLocation } from "react-router";
import { useLlmConfigured } from "#/hooks/use-llm-configured";
import { OnboardingModal } from "./onboarding-modal";
import {
  isOnboardingPreviewActive,
  readOnboardingPreviewStep,
} from "./onboarding-preview";

/**
 * Mounts the onboarding modal automatically when the active backend does not
 * yet have a usable LLM configuration.
 *
 * A configured active backend may provide everything onboarding would collect,
 * so the modal stays hidden once the backend already has a usable LLM.
 *
 * With `?previewOnboardingStep=<0-3>` the modal opens on that slide for
 * design review without persisting completion (works on any route when
 * mounted from the root layout).
 */
export function OnboardingHost() {
  const location = useLocation();
  const previewStep = readOnboardingPreviewStep(location.search);
  const isPreview = isOnboardingPreviewActive(location.search);
  const { isConfigured, isLoading } = useLlmConfigured();

  if (!isPreview) {
    if (isLoading) return null;
    if (isConfigured) return null;
  }

  const handleClose = () => {
    if (isPreview) return;
  };

  return (
    <OnboardingModal
      onClose={handleClose}
      initialStep={previewStep ?? 0}
      isPreview={isPreview}
    />
  );
}
