import React from "react";
import { HttpError, isAskUserAnswerEvent } from "@openhands/typescript-client";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { Check } from "lucide-react";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import {
  experimentalFeaturesQueryKey,
  useIsExperimentalFeatureEnabled,
} from "#/hooks/query/use-experimental-features";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import { useEventStore, type OHEvent } from "#/stores/use-event-store";
import type { ActionEvent, ObservationEvent } from "#/types/agent-server/core";
import type { AskUserAction } from "#/types/agent-server/core/base/action";
import {
  isMessageEvent,
  isObservationEvent,
} from "#/types/agent-server/type-guards";
import { formControlMultilineFieldClassName } from "#/utils/form-control-classes";
import { cn } from "#/utils/utils";

export function DecisionCard({
  question,
  result,
  events,
}: {
  question: ActionEvent<AskUserAction>;
  result: ObservationEvent & { isFromPlanningAgent?: boolean };
  events: readonly OHEvent[];
}) {
  const { t } = useTranslation("openhands");
  const { conversationId } = useOptionalConversationId();
  const { data: conversation } = useActiveConversation();
  const { backend } = useActiveBackend();
  const enabled = useIsExperimentalFeatureEnabled("ask_user");
  const queryClient = useQueryClient();
  const [text, setText] = React.useState("");
  const [sending, setSending] = React.useState(false);
  const [errorKey, setErrorKey] = React.useState<I18nKey>();
  const [saved, setSaved] = React.useState(false);
  const inFlight = React.useRef(false);
  const textId = React.useId();
  const questionId = React.useId();
  const planner = !!result.isFromPlanningAgent;
  const following = events.slice(
    events.findIndex((event) => event.id === result.id) + 1,
  );
  const closing = following.find(
    (event) =>
      !!event.isFromPlanningAgent === planner &&
      (("kind" in event && isAskUserAnswerEvent(event)) ||
        (isMessageEvent(event) && event.source === "user" && !event.sender) ||
        (isObservationEvent(event) &&
          event.observation.kind === "AskUserObservation" &&
          !event.observation.is_error)),
  );
  const answer =
    closing &&
    "kind" in closing &&
    isAskUserAnswerEvent(closing) &&
    closing.ask_id === question.id
      ? closing
      : undefined;
  const answeredInChat = closing && isMessageEvent(closing);
  const pending = !closing && !saved;
  const canAnswer = pending && enabled && !planner && !!conversationId;
  const action = question.action;

  const submit = async (optionIndex?: number) => {
    if (
      !canAnswer ||
      inFlight.current ||
      (optionIndex === undefined && !text.trim())
    )
      return;
    const store = useEventStore.getState();
    const scope = store.loadedConversationScope;
    inFlight.current = true;
    setSending(true);
    setErrorKey(undefined);
    try {
      const answerEvent = await AgentServerConversationService.answerAskUser(
        conversationId,
        question.id,
        {
          ...(optionIndex === undefined ? {} : { optionIndex }),
          ...(text.trim() ? { text: text.trim() } : {}),
        },
        conversation?.id === conversationId
          ? {
              conversationUrl: conversation.conversation_url ?? null,
              sessionApiKey: conversation.session_api_key ?? null,
            }
          : undefined,
      );
      const current = useEventStore.getState();
      if (
        current.loadedConversationId === conversationId &&
        current.loadedConversationScope === scope
      ) {
        current.addEvent(answerEvent);
      }
    } catch (error) {
      const capacity = error instanceof HttpError && error.status === 429;
      setSaved(capacity);
      setErrorKey(
        capacity
          ? I18nKey.ASK_USER$SAVED_NOT_RUNNING
          : I18nKey.ASK_USER$SEND_FAILED,
      );
      void queryClient.invalidateQueries({
        queryKey: ["conversation-history", conversationId],
      });
      if (error instanceof HttpError && error.status === 409) {
        void queryClient.invalidateQueries({
          queryKey: experimentalFeaturesQueryKey(backend),
        });
      }
    } finally {
      inFlight.current = false;
      setSending(false);
    }
  };

  return (
    <section
      aria-labelledby={questionId}
      data-testid="decision-card"
      className="mt-4 min-w-0 rounded-xl border border-border bg-surface p-4 text-sm text-contrast"
    >
      <h3 id={questionId} className="mb-3 font-medium wrap-anywhere">
        {action.question}
      </h3>
      {canAnswer ? (
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void submit();
          }}
          className="flex flex-col gap-3"
          aria-busy={sending}
        >
          <fieldset
            disabled={sending}
            aria-labelledby={questionId}
            className="flex min-w-0 flex-col gap-2"
          >
            {action.options.map((option, index) => (
              <button
                key={index}
                type="button"
                onClick={() => void submit(index)}
                className="min-w-0 cursor-pointer rounded-lg border border-control p-3 text-start focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus hover:bg-interactive-hover disabled:cursor-wait disabled:opacity-60"
              >
                <span className="min-w-0 wrap-anywhere">
                  <span className="font-medium">{option.label}</span>
                  {action.recommended === index && (
                    <span className="ml-2 text-xs text-muted">
                      {" "}
                      {t(I18nKey.ASK_USER$RECOMMENDED)}
                    </span>
                  )}
                  {option.consequence && (
                    <span className="mt-1 block text-muted">
                      {option.consequence}
                    </span>
                  )}
                </span>
              </button>
            ))}
          </fieldset>
          <div className="flex flex-col gap-1.5">
            <label htmlFor={textId} className="text-muted">
              {t(I18nKey.ASK_USER$YOUR_WORDS)}
            </label>
            <textarea
              id={textId}
              rows={2}
              value={text}
              disabled={sending}
              onChange={(event) => setText(event.target.value)}
              className={formControlMultilineFieldClassName}
            />
          </div>
          <BrandButton
            variant="primary"
            type="submit"
            isDisabled={sending || !text.trim()}
            aria-busy={sending}
            className="focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
          >
            {t(I18nKey.ASK_USER$SEND)}
          </BrandButton>
        </form>
      ) : (
        <div className="flex flex-col gap-2">
          {action.options.map((option, index) => (
            <div
              key={index}
              className={cn(
                "wrap-anywhere",
                answer?.option_index === index ? "font-medium" : "text-muted",
              )}
            >
              {answer?.option_index === index && (
                <Check className="mr-1 inline h-4 w-4" aria-hidden="true" />
              )}
              {option.label}
            </div>
          ))}
          {answer?.text && (
            <p className="whitespace-pre-wrap wrap-anywhere">{answer.text}</p>
          )}
          <p className="text-muted" role="status">
            {t(
              answer || saved
                ? I18nKey.ASK_USER$ANSWERED
                : answeredInChat
                  ? I18nKey.ASK_USER$ANSWERED_IN_CHAT
                  : closing
                    ? I18nKey.ASK_USER$SUPERSEDED
                    : planner
                      ? I18nKey.ASK_USER$PLANNER
                      : I18nKey.ASK_USER$DISABLED,
            )}
          </p>
        </div>
      )}
      {errorKey && (
        <p role="alert" className="mt-3 text-feedback-error">
          {t(errorKey)}
        </p>
      )}
    </section>
  );
}
