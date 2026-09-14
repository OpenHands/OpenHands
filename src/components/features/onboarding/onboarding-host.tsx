import { useLocation } from "react-router";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { SEEDED_DEFAULT_BACKEND_ID } from "#/api/backend-registry/default-backend";
import { useSettings } from "#/hooks/query/use-settings";
import { isSubscriptionLlmConfig } from "#/constants/llm-subscription";
import type { Settings } from "#/types/settings";
import { OnboardingModal } from "./onboarding-modal";
import {
  isOnboardingPreviewActive,
  readOnboardingPreviewStep,
} from "./onboarding-preview";
import { useOnboardingCompletion } from "./use-onboarding-completion";

function hasUsableLlm(settings: Settings | undefined): boolean {
  const llm = settings?.agent_settings?.llm as
    | Record<string, unknown>
    | undefined;
  const hasModel =
    typeof llm?.model === "string" && llm.model.trim().length > 0;
  const hasAuth =
    settings?.llm_api_key_set === true || isSubscriptionLlmConfig(llm);
  return hasModel && hasAuth;
}

/**
 * Mounts the onboarding modal automatically the first time the user
 * lands on a host route (i.e. when the localStorage onboarding flag
 * isn't set yet). Closing or completing the flow marks it done so the
 * modal won't re-appear on subsequent visits.
 *
 * A backend already reporting a usable LLM makes the modal redundant, so it is
 * skipped — except the launcher-seeded default-local one, which may hold an
 * LLM from an earlier user, leaving `openhands-onboarded` its first-run signal.
 *
 * With `?previewOnboardingStep=<0-3>` the modal opens on that slide for
 * design review without persisting completion (works on any route when
 * mounted from the root layout).
 */
export function OnboardingHost() {
  const location = useLocation();
  const previewStep = readOnboardingPreviewStep(location.search);
  const isPreview = isOnboardingPreviewActive(location.search);
  const { isCompleted, markCompleted } = useOnboardingCompletion();
  const { backend } = useActiveBackend();
  const settings = useSettings();
  // A backend the user pointed at explicitly (Cloud, or a Local server added
  // via "Add Backend") is one whose configuration this browser did not create,
  // so its reported LLM is trustworthy evidence that setup is already done.
  const trustsBackendLlm =
    backend.kind === "cloud" || backend.id !== SEEDED_DEFAULT_BACKEND_ID;

  if (!isPreview) {
    if (isCompleted) return null;
    if (trustsBackendLlm && settings.isLoading) return null;
    if (trustsBackendLlm && hasUsableLlm(settings.data)) {
      return null;
    }
  }

  const handleClose = () => {
    if (isPreview) return;
    markCompleted();
  };

  return (
    <OnboardingModal
      onClose={handleClose}
      initialStep={previewStep ?? 0}
      isPreview={isPreview}
    />
  );
}
