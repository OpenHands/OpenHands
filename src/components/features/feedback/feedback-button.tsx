import { useState } from "react";
import { Info, MessageSquare } from "lucide-react";
import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import {
  ModalBody,
  MODAL_MAX_WIDTH_VIEWPORT,
} from "#/components/shared/modals/modal-body";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import { submitPrivateFeedback } from "#/services/telemetry";

export function FeedbackButton() {
  const { t } = useTranslation("openhands");
  const [open, setOpen] = useState(false);
  const [showDisclosure, setShowDisclosure] = useState(false);
  const [message, setMessage] = useState("");
  const [status, setStatus] = useState<"idle" | "sending" | "error" | "sent">(
    "idle",
  );

  const close = () => {
    if (status !== "sending") {
      setOpen(false);
      setShowDisclosure(false);
      if (status === "sent") {
        setStatus("idle");
      }
    }
  };
  const submit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const text = message.trim();
    if (!text || status === "sending") return;
    setStatus("sending");
    try {
      await submitPrivateFeedback(text);
      setMessage("");
      setStatus("sent");
    } catch {
      setStatus("error");
    }
  };

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="inline-flex items-center gap-2 rounded-md px-3 py-2 text-sm text-muted hover:bg-surface-raised hover:text-contrast"
      >
        <MessageSquare aria-hidden size={16} />
        {t(I18nKey.FEEDBACK$TITLE)}
      </button>
      {open && (
        <ModalBackdrop onClose={close} aria-label={t(I18nKey.FEEDBACK$TITLE)}>
          <ModalBody
            className={`relative ${MODAL_MAX_WIDTH_VIEWPORT} items-stretch gap-4 border border-border`}
          >
            <ModalCloseButton onClose={close} disabled={status === "sending"} />
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-semibold text-contrast">
                {t(I18nKey.FEEDBACK$TITLE)}
              </h2>
              <StyledTooltip
                content={t(I18nKey.FEEDBACK$CONSENT)}
                placement="bottom"
                isOpen={showDisclosure}
              >
                <button
                  type="button"
                  aria-label={t(I18nKey.FEEDBACK$CONSENT)}
                  className="inline-flex rounded text-muted hover:text-contrast"
                  onFocus={() => setShowDisclosure(true)}
                  onBlur={() => setShowDisclosure(false)}
                  onMouseEnter={() => setShowDisclosure(true)}
                  onMouseLeave={() => setShowDisclosure(false)}
                  onClick={() => setShowDisclosure(true)}
                >
                  <Info aria-hidden size={16} />
                </button>
              </StyledTooltip>
            </div>
            {status === "sent" ? (
              <p role="status" className="text-sm text-feedback-success">
                {t(I18nKey.FEEDBACK$SUCCESS)}
              </p>
            ) : (
              <form onSubmit={submit} className="flex flex-col gap-3">
                <label
                  htmlFor="feedback-message"
                  className="text-sm text-contrast"
                >
                  {t(I18nKey.FEEDBACK$QUESTION)}
                </label>
                <textarea
                  id="feedback-message"
                  required
                  value={message}
                  onChange={(event) => setMessage(event.target.value)}
                  rows={5}
                  className="w-full resize-y rounded-md border border-border bg-base-secondary p-2 text-contrast"
                />
                <p className="text-xs text-muted">
                  {t(I18nKey.FEEDBACK$PRIVACY)}
                </p>
                {status === "error" && (
                  <p role="alert" className="text-sm text-feedback-error">
                    {t(I18nKey.FEEDBACK$ERROR)}
                  </p>
                )}
                <BrandButton
                  type="submit"
                  variant="primary"
                  isDisabled={!message.trim() || status === "sending"}
                >
                  {t(I18nKey.FEEDBACK$SEND)}
                </BrandButton>
              </form>
            )}
          </ModalBody>
        </ModalBackdrop>
      )}
    </>
  );
}
