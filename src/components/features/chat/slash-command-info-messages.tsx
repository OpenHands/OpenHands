import { useSlashCommandInfoStore } from "#/stores/slash-command-info-store";
import { GenericEventMessage } from "./generic-event-message";
import { GotItButton } from "./got-it-button";

export interface SlashCommandInfoMessagesProps {
  conversationId: string | null | undefined;
}

/**
 * Renders the rendered-in-chat answers produced by display-only slash
 * commands (/help, /skills), mirroring BtwMessages.
 */
export function SlashCommandInfoMessages({
  conversationId,
}: SlashCommandInfoMessagesProps) {
  const entriesById = useSlashCommandInfoStore((s) => s.entriesByConversation);
  const dismiss = useSlashCommandInfoStore((s) => s.dismiss);
  const entries = conversationId ? (entriesById[conversationId] ?? []) : [];

  if (!conversationId || entries.length === 0) return null;

  return (
    <div
      data-testid="slash-command-info-messages"
      className="custom-scrollbar-always flex max-h-[40vh] w-full flex-col overflow-y-auto overscroll-contain"
    >
      {entries.map((entry) => (
        <GenericEventMessage
          key={entry.id}
          title={
            <span className="flex items-center gap-2">
              <span className="opacity-60">{entry.title}</span>
            </span>
          }
          details={entry.body}
          initiallyExpanded
          chevronPosition="before"
          titleTrailing={
            <GotItButton onClick={() => dismiss(conversationId, entry.id)} />
          }
        />
      ))}
    </div>
  );
}
