import React from "react";
import { useTranslation } from "react-i18next";

import { getLockedCloudHost } from "#/api/agent-server-config";
import { BrandButton } from "#/components/features/settings/brand-button";
import { useActiveBackend } from "#/contexts/active-backend-context";
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
 * Hidden on hosted installs, gated on the same pair of signals as
 * `telemetry-consent-banner`: locked-to-Cloud, or a non-local active backend.
 *
 * ## Telemetry
 *
 * | name | properties |
 * | --- | --- |
 * | `canvas_feedback_submitted` | `feedback`, `feedback_length`, `has_email`, `conversation_id`, plus the common backend context |
 *
 * A supplied email is attached to the PostHog person with
 * `setPersonProperties` as the `email` person property, not repeated on the
 * event.
 */

/** Deliberately permissive: reject obvious typos, not unusual-but-valid addresses. */
const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

type SubmitState = "idle" | "submitting" | "submitted" | "error";

export function FeedbackLauncher() {
  const { t } = useTranslation();
  const { trackFeedbackSubmitted, attachFeedbackEmail } = useTracking();
  const { conversationId } = useOptionalConversationId();
  const { backend } = useActiveBackend();

  const [isOpen, setIsOpen] = React.useState(false);
  const [feedback, setFeedback] = React.useState("");
  const [email, setEmail] = React.useState("");
  const [emailError, setEmailError] = React.useState(false);
  const [state, setState] = React.useState<SubmitState>("idle");
  const triggerRef = React.useRef<HTMLButtonElement>(null);

  // Both signals, matching `telemetry-consent-banner` — the other install-gated
  // control mounted alongside this one. `getLockedCloudHost()` alone is not
  // enough: a self-hosted OHE reached on its own domain is not locked to Cloud,
  // and `backend-form-modal` notes it is otherwise indistinguishable from a
  // local agent-server by host.
  const isHostedInstall =
    getLockedCloudHost() !== null || backend.kind !== "local";

  const close = React.useCallback(() => {
    setIsOpen(false);
    // Reset here rather than only on the success button, so reopening never
    // shows a stale "thank you" or error from the previous submission.
    setState("idle");
    setEmailError(false);
  }, []);

  React.useEffect(() => {
    if (!isOpen) return undefined;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      close();
      triggerRef.current?.focus();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [isOpen, close]);

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

  if (isHostedInstall) return null;

  return (
    <div className="fixed bottom-4 right-4 z-30 flex flex-col items-end gap-2">
      {isOpen && (
        <div
          data-testid="feedback-panel"
          role="dialog"
          aria-modal="false"
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
                onClick={close}
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
                  onClick={close}
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
                    : t(I18nKey.BUTTON$SEND)}
                </BrandButton>
              </div>
            </form>
          )}
        </div>
      )}

      <BrandButton
        ref={triggerRef}
        type="button"
        variant="primary"
        testId="feedback-launcher"
        ariaLabel={t(I18nKey.FEEDBACK$TITLE)}
        aria-haspopup="dialog"
        aria-expanded={isOpen}
        onClick={() => (isOpen ? close() : setIsOpen(true))}
      >
        {t(I18nKey.FEEDBACK$TITLE)}
      </BrandButton>
    </div>
  );
}
