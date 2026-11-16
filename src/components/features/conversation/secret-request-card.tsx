import React from "react";
import { useTranslation } from "react-i18next";
import { SecretForm } from "#/components/features/settings/secrets-settings/secret-form";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { I18nKey } from "#/i18n/declaration";
import { completeSecretRequest } from "#/services/secret-request";
import {
  secretRequestMutationKey,
  type SecretRequest,
  useSecretRequestDraftStore,
  useSecretRequestStore,
} from "#/stores/secret-request-store";

/** The current conversation's pending secret requests, oldest first. */
export function usePendingSecretRequests(): SecretRequest[] {
  const { conversationId } = useOptionalConversationId();
  const requests = useSecretRequestStore((s) => s.requests);
  return React.useMemo(
    () => requests.filter((r) => r.conversationId === conversationId),
    [requests, conversationId],
  );
}

/**
 * Which field of each pending request's card held focus, so a remount can put
 * the caret back. The card moves from its row into the trailing content when the
 * list crosses into virtualization, and the whole tree is replaced, so React
 * remounts the form: the draft store keeps the text, but nothing restores the
 * caret, and a reader mid-secret loses the field they were filling (the agent
 * keeps working, so the 150th row can land while they type).
 *
 * Updated on focusin/focusout rather than at unmount, because React detaches
 * the section's ref before the card's own effect cleanup runs, so the cleanup
 * cannot read the DOM it is unmounting. Keyed by tool call id and held outside
 * React so it survives the remount.
 */
const focusedFieldByToolCall = new Map<string, string>();

/**
 * Collects a secret the agent asked for with `request_secret`, as a card in
 * the chat. It does not block the conversation: the agent keeps working and
 * the user answers whenever they are ready. The value goes straight to the
 * secrets store; the agent only learns the outcome.
 */
export function SecretRequestCard({ request }: { request: SecretRequest }) {
  const { t } = useTranslation("openhands");
  const draft = useSecretRequestDraftStore((s) => s.drafts[request.toolCallId]);
  const setDraft = useSecretRequestDraftStore((s) => s.setDraft);
  const sectionRef = React.useRef<HTMLElement>(null);

  // Runs after `SecretForm`'s own effects (children before parents), so the
  // restored input exists by the time this focuses it.
  React.useLayoutEffect(() => {
    const field = focusedFieldByToolCall.get(request.toolCallId);
    if (!field) return;
    sectionRef.current
      ?.querySelector<HTMLInputElement>(`input[name="${field}"]`)
      ?.focus();
  }, [request.toolCallId]);

  // A focusout that lands outside the card (the reader moved to the composer)
  // clears the entry, so the remount does not steal focus back. Focus moving
  // between the card's own fields keeps it.
  const handleBlur = (event: React.FocusEvent<HTMLElement>) => {
    const next = event.relatedTarget;
    if (next instanceof Node && sectionRef.current?.contains(next)) return;
    focusedFieldByToolCall.delete(request.toolCallId);
  };

  return (
    <section
      ref={sectionRef}
      data-testid="secret-request-card"
      onFocus={(event) => {
        const { name } = event.target as HTMLInputElement;
        if (name) focusedFieldByToolCall.set(request.toolCallId, name);
      }}
      onBlur={handleBlur}
      aria-label={t(I18nKey.SECRETS$REQUEST_TITLE)}
      className="mt-4 flex flex-col gap-3 rounded-xl border border-border bg-surface p-4"
    >
      <div className="flex flex-col gap-1">
        <span className="text-sm font-medium text-contrast">
          {t(I18nKey.SECRETS$REQUEST_TITLE)}
        </span>
        {request.reason ? (
          <span className="text-sm text-muted">{request.reason}</span>
        ) : null}
      </div>
      {/* SecretForm also calls onCancel after a successful save; the request
          is already complete by then, so that call is a no-op. The draft is
          held in the store so a remount across the virtualization threshold
          (the card moving to the trailing list) keeps what the user typed, and
          the mutation key lets the form read its in-flight state from the cache
          entry so a remount cannot re-enable Save mid-request. */}
      <SecretForm
        mode="add"
        selectedSecret={null}
        defaultName={request.name}
        keepOpenOnError
        initialDraft={draft}
        mutationKey={secretRequestMutationKey(request.toolCallId)}
        onDraftChange={(next) => setDraft(request.toolCallId, next)}
        onSaved={(secret) => void completeSecretRequest(request, secret)}
        onCancel={() => void completeSecretRequest(request, null)}
      />
    </section>
  );
}
