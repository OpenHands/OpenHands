import React from "react";
import { useTranslation } from "react-i18next";
import { CheckCircle2, HelpCircle, Sparkles } from "lucide-react";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { I18nKey } from "#/i18n/declaration";
import {
  type AskUserDecisionRequest,
  useAskUserStore,
} from "#/stores/ask-user-store";
import { completeAskUserDecision } from "#/services/ask-user";
import { cn } from "#/utils/utils";

/** The current conversation's ask_user decision requests, oldest first. */
export function usePendingAskUserRequests(): AskUserDecisionRequest[] {
  const { conversationId } = useOptionalConversationId();
  const requests = useAskUserStore((s) => s.requests);
  return React.useMemo(
    () => requests.filter((r) => r.conversationId === conversationId),
    [requests, conversationId],
  );
}

interface AskUserCardProps {
  request: AskUserDecisionRequest;
}

/**
 * Presents an agent's question as an interactive decision card with selectable
 * options, recommended badges, custom response input, and settled outcome state.
 */
export function AskUserCard({ request }: AskUserCardProps) {
  const { t } = useTranslation("openhands");
  const [selectedOption, setSelectedOption] = React.useState<string | null>(
    request.recommended ?? request.options[0]?.label ?? null,
  );
  const [customAnswer, setCustomAnswer] = React.useState("");
  const [isSubmitting, setIsSubmitting] = React.useState(false);

  const isAnswered = !!request.selectedAnswer;

  const handleSubmit = async () => {
    const answer = customAnswer.trim() || selectedOption;
    if (!answer || isSubmitting) return;

    setIsSubmitting(true);
    try {
      await completeAskUserDecision(request, answer);
    } finally {
      setIsSubmitting(false);
    }
  };

  if (isAnswered) {
    return (
      <section
        data-testid="ask-user-card-settled"
        aria-label={t(I18nKey.ASK_USER$CARD_TITLE)}
        className="mt-4 flex flex-col gap-2 rounded-xl border border-border-subtle bg-surface/70 p-4"
      >
        <div className="flex items-center gap-2 text-xs font-medium text-muted">
          <HelpCircle size={14} className="shrink-0 text-primary" aria-hidden />
          <span>{t(I18nKey.ASK_USER$CARD_TITLE)}</span>
        </div>
        <p className="text-sm font-medium text-contrast">{request.question}</p>
        <div className="mt-1 flex items-center gap-2 rounded-lg border border-border-subtle bg-tertiary px-3 py-2 text-xs text-contrast">
          <CheckCircle2
            size={14}
            className="shrink-0 text-primary"
            aria-hidden
          />
          <span className="font-medium text-muted">
            {t(I18nKey.ASK_USER$ANSWERED_PREFIX)}
          </span>
          <span className="font-semibold text-contrast">
            {request.selectedAnswer}
          </span>
        </div>
      </section>
    );
  }

  return (
    <section
      data-testid="ask-user-card"
      aria-label={t(I18nKey.ASK_USER$CARD_TITLE)}
      className="mt-4 flex flex-col gap-3 rounded-xl border border-border bg-surface p-4 shadow-sm"
    >
      <div className="flex items-center gap-2">
        <HelpCircle size={16} className="shrink-0 text-primary" aria-hidden />
        <span className="text-xs font-semibold uppercase tracking-wider text-primary">
          {t(I18nKey.ASK_USER$CARD_TITLE)}
        </span>
      </div>

      <p className="text-sm font-medium text-contrast leading-relaxed">
        {request.question}
      </p>

      {request.options.length > 0 && (
        <div className="flex flex-col gap-2 pt-1">
          {request.options.map((option, index) => {
            const isSelected =
              selectedOption === option.label && !customAnswer.trim();
            const isRecommended = request.recommended === option.label;

            return (
              <button
                key={option.label || index}
                type="button"
                data-testid={`ask-user-option-${index}`}
                onClick={() => {
                  setSelectedOption(option.label);
                  setCustomAnswer("");
                }}
                className={cn(
                  "flex flex-col items-start gap-1 rounded-lg border p-3 text-left transition-colors cursor-pointer",
                  isSelected
                    ? "border-primary bg-primary/10 shadow-xs"
                    : "border-border-subtle bg-tertiary hover:border-border hover:bg-tertiary/80",
                )}
              >
                <div className="flex w-full items-center justify-between gap-2">
                  <span
                    className={cn(
                      "text-sm font-medium",
                      isSelected ? "text-primary" : "text-contrast",
                    )}
                  >
                    {option.label}
                  </span>
                  {isRecommended && (
                    <span className="inline-flex items-center gap-1 rounded-full bg-primary/20 px-2 py-0.5 text-xs font-medium text-primary">
                      <Sparkles size={11} aria-hidden />
                      {t(I18nKey.ASK_USER$RECOMMENDED)}
                    </span>
                  )}
                </div>
                {option.description && (
                  <p className="text-xs text-muted leading-normal">
                    {option.description}
                  </p>
                )}
              </button>
            );
          })}
        </div>
      )}

      <div className="flex flex-col gap-2 pt-1">
        <input
          type="text"
          data-testid="ask-user-custom-input"
          value={customAnswer}
          onChange={(e) => setCustomAnswer(e.target.value)}
          placeholder={t(I18nKey.ASK_USER$CUSTOM_ANSWER_PLACEHOLDER)}
          className="w-full rounded-md border border-border-subtle bg-tertiary px-3 py-2 text-xs text-contrast placeholder:text-muted focus:border-primary focus:outline-hidden"
        />

        <div className="flex justify-end pt-1">
          <button
            type="button"
            data-testid="ask-user-submit"
            disabled={(!selectedOption && !customAnswer.trim()) || isSubmitting}
            onClick={handleSubmit}
            className={cn(
              "inline-flex items-center justify-center rounded-md px-4 py-2 text-xs font-medium transition-colors cursor-pointer",
              (!selectedOption && !customAnswer.trim()) || isSubmitting
                ? "cursor-not-allowed opacity-50 bg-muted/20 text-muted"
                : "bg-primary text-primary-foreground hover:bg-primary/90",
            )}
          >
            {t(I18nKey.ASK_USER$SUBMIT_ANSWER)}
          </button>
        </div>
      </div>
    </section>
  );
}
