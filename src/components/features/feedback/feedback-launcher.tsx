import React from "react";
import { useTranslation } from "react-i18next";

import { getLockedCloudHost } from "#/api/agent-server-config";
import { BrandButton } from "#/components/features/settings/brand-button";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useTracking } from "#/hooks/use-tracking";
import { I18nKey } from "#/i18n/declaration";
import { isTelemetryEnabled } from "#/services/telemetry";
import { cn } from "#/utils/utils";

/**
 * Persistent feedback control for non-OHE installs.
 *
 * Renders a floating button in the bottom-right that opens a small panel. It is
 * deliberately not a popup: the panel only ever opens from a user click, so the
 * control cannot become a recurring interruption. The button is the anchor a
 * future survey bubble would attach to.
 *
 * Hidden entirely when the app is locked to Cloud — the same signal
 * `agent-canvas-update-card` uses to hide install-specific affordances on a
 * hosted deployment.
 *
 * ## Telemetry
 *
 * | name | properties |
 * | --- | --- |
 * | `canvas_feedback_submitted` | `feedback`, `feedback_length`, `has_email`, `conversation_id`, plus the common backend context |
 *
 * A supplied email is attached to the PostHog person via `identify` as the
 * `email` person property, not repeated on the event.
 */

/** Deliberately permissive: reject obvious typos, not unusual-but-valid addresses. */
const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

type SubmitState = "idle" | "submitting" | "submitted" | "error";

export function FeedbackLauncher() {
  const { t } = useTranslation();
  const { trackFeedbackSubmitted, attachFeedbackEmail } = useTracking();
  const { conversationId } = useOptionalConversationId();

  const [isOpen, setIsOpen] = React.useState(false);
  const [feedback, setFeedback] = React.useState("");
  const [email, setEmail] = React.useState("");
  const [emailError, setEmailError] = React.useState(false);
  const [state, setState] = React.useState<SubmitState>("idle");

  // Not a hook, so it is safe to read on every render; the early return below
  // happens after every hook has run.
  const isLockedToCloud = getLockedCloudHost() !== null;

  /**
   * The form is `noValidate`: the browser's own constraint messages are not
   * translated and differ per engine, so validation is owned here and reported
   * through the i18n catalogue instead.
   */
  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();

    const trimmedFeedback = feedback.trim();
    const trimmedEmail = email.trim();
    if (!trimmedFeedback) return;

    if (trimmedEmail && !EMAIL_PATTERN.test(trimmedEmail)) {
      setEmailError(true);
      return;
    }
    setEmailError(false);

    // Consent is the one failure this can detect up front. `trackEvent`
    // resolves without capturing when consent is absent, so a resolved promise
    // is not evidence the feedback landed.
    if (!isTelemetryEnabled()) {
      setState("error");
      return;
    }

    setState("submitting");
    try {
      if (trimmedEmail) await attachFeedbackEmail(trimmedEmail);
      await trackFeedbackSubmitted({
        feedback: trimmedFeedback,
        hasEmail: Boolean(trimmedEmail),
        conversationId: conversationId ?? undefined,
      });
      setState("submitted");
      setFeedback("");
      setEmail("");
    } catch {
      // Keep what the user typed so a retry does not start from scratch.
      setState("error");
    }
  };

  if (isLockedToCloud) return null;

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col items-end gap-2">
      {isOpen && (
        <div
          data-testid="feedback-panel"
          role="dialog"
          aria-label={t(I18nKey.FEEDBACK$TITLE)}
          className="w-[min(20rem,calc(100vw-2rem))] rounded-lg border border-[var(--oh-border-subtle)] bg-tertiary p-4 shadow-lg"
        >
          {state === "submitted" ? (
            <div className="flex flex-col gap-3">
              <p data-testid="feedback-success" className="text-sm text-white">
                {t(I18nKey.FEEDBACK$THANK_YOU_FOR_FEEDBACK)}
              </p>
              <BrandButton
                type="button"
                variant="secondary"
                testId="feedback-close"
                onClick={() => {
                  setIsOpen(false);
                  setState("idle");
                }}
              >
                {t(I18nKey.BUTTON$CLOSE)}
              </BrandButton>
            </div>
          ) : (
            <form
              className="flex flex-col gap-3"
              onSubmit={handleSubmit}
              noValidate
            >
              <p className="text-sm text-white">
                {t(I18nKey.FEEDBACK$DESCRIPTION)}
              </p>

              <label className="flex flex-col gap-1 text-xs text-white">
                {t(I18nKey.FEEDBACK$TITLE)}
                <textarea
                  data-testid="feedback-message"
                  value={feedback}
                  onChange={(e) => setFeedback(e.target.value)}
                  rows={4}
                  required
                  className="rounded border border-[var(--oh-border-input)] bg-base p-2 text-sm text-white"
                />
              </label>

              <label className="flex flex-col gap-1 text-xs text-white">
                <span>
                  {t(I18nKey.FEEDBACK$EMAIL_LABEL)} (
                  {t(I18nKey.COMMON$OPTIONAL)})
                </span>
                <input
                  data-testid="feedback-email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder={t(I18nKey.FEEDBACK$EMAIL_PLACEHOLDER)}
                  aria-invalid={emailError}
                  className={cn(
                    "rounded border bg-base p-2 text-sm text-white",
                    emailError
                      ? "border-danger"
                      : "border-[var(--oh-border-input)]",
                  )}
                />
              </label>

              {emailError && (
                <p
                  data-testid="feedback-email-error"
                  className="text-xs text-danger"
                >
                  {t(I18nKey.FEEDBACK$INVALID_EMAIL_FORMAT)}
                </p>
              )}

              {state === "error" && (
                <p data-testid="feedback-error" className="text-xs text-danger">
                  {t(I18nKey.FEEDBACK$FAILED_TO_SUBMIT)}
                </p>
              )}

              <div className="flex justify-end gap-2">
                <BrandButton
                  type="button"
                  variant="secondary"
                  testId="feedback-cancel"
                  onClick={() => setIsOpen(false)}
                >
                  {t(I18nKey.FEEDBACK$CANCEL_LABEL)}
                </BrandButton>
                <BrandButton
                  type="submit"
                  variant="primary"
                  testId="feedback-submit"
                  isDisabled={state === "submitting" || !feedback.trim()}
                  aria-busy={state === "submitting"}
                >
                  {state === "submitting"
                    ? t(I18nKey.FEEDBACK$SUBMITTING_LABEL)
                    : t(I18nKey.FEEDBACK$SHARE_LABEL)}
                </BrandButton>
              </div>
            </form>
          )}
        </div>
      )}

      <BrandButton
        type="button"
        variant="primary"
        testId="feedback-launcher"
        ariaLabel={t(I18nKey.FEEDBACK$TITLE)}
        onClick={() => setIsOpen((open) => !open)}
      >
        {t(I18nKey.FEEDBACK$TITLE)}
      </BrandButton>
    </div>
  );
}
