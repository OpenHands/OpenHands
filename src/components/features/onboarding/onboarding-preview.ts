export const ONBOARDING_PREVIEW_STEP_QUERY_PARAM = "previewOnboardingStep";

const MAX_ONBOARDING_STEP = 3;

/** Reads `?previewOnboardingStep=<0-3>` for dev/design review of a single slide. */
export function readOnboardingPreviewStep(
  search = typeof window !== "undefined" ? window.location.search : "",
): number | null {
  const raw = new URLSearchParams(search).get(
    ONBOARDING_PREVIEW_STEP_QUERY_PARAM,
  );
  if (raw == null) return null;

  const step = Number.parseInt(raw, 10);
  if (!Number.isFinite(step) || step < 0 || step > MAX_ONBOARDING_STEP) {
    return null;
  }
  return step;
}

export function isOnboardingPreviewActive(
  search = typeof window !== "undefined" ? window.location.search : "",
): boolean {
  return readOnboardingPreviewStep(search) != null;
}
