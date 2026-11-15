import React from "react";
import { useTranslation } from "react-i18next";
import { useMutation } from "@tanstack/react-query";
import { Check } from "lucide-react";
import { askUserService } from "#/api/ask-user-service/ask-user-service.api";
import { I18nKey } from "#/i18n/declaration";
import { displayApiErrorToast } from "#/utils/custom-toast-handlers";
import { formControlMultilineFieldClassName } from "#/utils/form-control-classes";
import { cn } from "#/utils/utils";
import type { AskUserCard as AskUserCardState } from "./ask-user-cards";

interface AskUserCardProps {
  card: AskUserCardState;
  /** The conversation the card answers in; without one it is read-only. */
  conversationId: string | null | undefined;
}

/**
 * A question the agent asked with `ask_user`, as a card in the chat. While it
 * waits, the user taps an option (optionally adding a note), or writes in the
 * chat instead; either answers it. Once answered it folds to the question and
 * the answer. The answer reaches the agent as the user's next message, so
 * nothing else needs to happen here.
 */
export function AskUserCard({ card, conversationId }: AskUserCardProps) {
  const { t } = useTranslation("openhands");
  const [note, setNote] = React.useState("");
  const [picked, setPicked] = React.useState<number | null>(null);
  const noteId = React.useId();
  const { status } = card;
  const pending = status.kind === "pending";

  const answer = useMutation({
    mutationFn: (optionIndex: number) =>
      askUserService.answer({
        conversationId: conversationId!,
        askId: card.askId,
        optionIndex,
        text: note.trim() || undefined,
      }),
    onError: (error) => {
      setPicked(null);
      displayApiErrorToast(error, t(I18nKey.ASK_USER$ANSWER_FAILED));
    },
  });
  const canAnswer = pending && !!conversationId && picked === null;

  return (
    <section
      data-testid="ask-user-card"
      data-status={status.kind}
      aria-label={t(I18nKey.ASK_USER$TITLE)}
      className="mt-4 flex flex-col gap-3 rounded-xl border border-border bg-surface p-4"
    >
      <div className="flex flex-col gap-1">
        <span className="text-xs font-medium uppercase tracking-wide text-muted">
          {t(I18nKey.ASK_USER$TITLE)}
        </span>
        <p className="text-sm font-medium text-contrast">{card.question}</p>
      </div>

      {pending ? (
        <>
          <ul className="flex flex-col gap-2">
            {card.options.map((option, index) => (
              <li key={`${index}-${option.label}`}>
                <button
                  type="button"
                  data-testid="ask-user-option"
                  disabled={!canAnswer}
                  aria-busy={picked === index}
                  onClick={() => {
                    setPicked(index);
                    answer.mutate(index);
                  }}
                  className={cn(
                    "flex w-full flex-col gap-0.5 rounded-lg border border-control bg-base-secondary px-3 py-2 text-left",
                    "transition-control-shell duration-75 motion-reduce:transition-none",
                    "hover:bg-surface-raised focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary",
                    "disabled:cursor-default disabled:hover:bg-base-secondary",
                    picked !== null && picked !== index && "opacity-60",
                    picked === index && "border-primary",
                  )}
                >
                  <span className="flex items-center gap-2 text-sm font-medium text-contrast">
                    {option.label}
                    {card.recommended === index ? (
                      <span
                        data-testid="ask-user-recommended"
                        className="rounded-full bg-interactive-hover px-2 py-0.5 text-xs font-normal text-muted"
                      >
                        {t(I18nKey.ASK_USER$RECOMMENDED)}
                      </span>
                    ) : null}
                  </span>
                  {option.consequence ? (
                    <span className="text-sm text-muted">
                      {option.consequence}
                    </span>
                  ) : null}
                </button>
              </li>
            ))}
          </ul>
          <label htmlFor={noteId} className="sr-only">
            {t(I18nKey.ASK_USER$NOTE_PLACEHOLDER)}
          </label>
          <textarea
            id={noteId}
            data-testid="ask-user-note"
            rows={1}
            value={note}
            disabled={!canAnswer}
            placeholder={t(I18nKey.ASK_USER$NOTE_PLACEHOLDER)}
            onChange={(event) => setNote(event.target.value)}
            className={cn(formControlMultilineFieldClassName, "resize-none")}
          />
          <span className="text-xs text-muted">
            {t(I18nKey.ASK_USER$CHAT_HINT)}
          </span>
        </>
      ) : (
        <AskUserOutcome card={card} />
      )}
    </section>
  );
}

function AskUserOutcome({ card }: { card: AskUserCardState }) {
  const { t } = useTranslation("openhands");
  const { status } = card;
  if (status.kind === "answered") {
    const label =
      status.optionLabel ??
      (status.optionIndex !== null
        ? card.options[status.optionIndex]?.label
        : null);
    return (
      <div data-testid="ask-user-outcome" className="flex flex-col gap-1">
        {label ? (
          <span className="flex items-center gap-2 text-sm text-contrast">
            <Check className="h-4 w-4 shrink-0 text-primary" aria-hidden />
            {label}
          </span>
        ) : null}
        {status.text ? (
          <p className="whitespace-pre-wrap text-sm text-muted">
            {status.text}
          </p>
        ) : null}
      </div>
    );
  }
  return (
    <span data-testid="ask-user-outcome" className="text-sm text-muted">
      {status.kind === "answered_in_chat"
        ? t(I18nKey.ASK_USER$ANSWERED_IN_CHAT)
        : t(I18nKey.ASK_USER$CLOSED)}
    </span>
  );
}
