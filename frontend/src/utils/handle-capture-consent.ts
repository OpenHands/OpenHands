import type { PostHog } from "posthog-js";

/**
 * Handle user consent for tracking
 * @param posthog PostHog instance (from usePostHog hook)
 * @param consent Whether the user consents to tracking
 */
export const handleCaptureConsent = (
  posthog: PostHog | undefined,
  consent: boolean,
) => {
  if (!posthog) return;

  if (consent && !posthog.has_opted_in_capturing()) {
    posthog.opt_in_capturing();
  }

  if (!consent && !posthog.has_opted_out_capturing()) {
    posthog.opt_out_capturing();
  }
};
