import CheckCircle from "#/icons/check-circle-solid.svg?react";
import { useBtwStore } from "#/stores/btw-store";
import { GenericEventMessage } from "./generic-event-message";

function GotItButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium text-success bg-success/10 hover:bg-success/20 border border-success/30 transition-colors"
    >
      <CheckCircle className="w-3.5 h-3.5 fill-success" />
      {/* eslint-disable-next-line i18next/no-literal-string */}
      <span>Got it</span>
    </button>
  );
}

export interface BtwMessagesProps {
  conversationId: string | null | undefined;
}

export function BtwMessages({ conversationId }: BtwMessagesProps) {
  const entriesById = useBtwStore((s) => s.entriesByConversation);
  const dismiss = useBtwStore((s) => s.dismiss);
  const entries = conversationId ? (entriesById[conversationId] ?? []) : [];

  if (!conversationId || entries.length === 0) return null;

  return (
    <div data-testid="btw-messages" className="flex flex-col w-full">
      {entries.map((entry) => {
        const isPending = entry.status === "pending";
        return (
          <GenericEventMessage
            key={entry.id}
            title={
              <span className="flex items-center gap-2">
                {/* eslint-disable-next-line i18next/no-literal-string */}
                <span className="opacity-60">BTW:</span>
                <span>{entry.question}</span>
                {isPending && (
                  <span
                    data-testid="btw-spinner"
                    className="inline-block w-3.5 h-3.5 ml-2 rounded-full border-2 border-neutral-500 border-t-transparent animate-spin"
                  />
                )}
              </span>
            }
            details={
              isPending
                ? "Waiting for the agent's answer…"
                : (entry.response ?? "")
            }
            initiallyExpanded={!isPending}
            chevronPosition="before"
            titleTrailing={
              !isPending && (
                <GotItButton
                  onClick={() => dismiss(conversationId, entry.id)}
                />
              )
            }
          />
        );
      })}
    </div>
  );
}
