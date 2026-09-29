import { useTranslation } from "react-i18next";
import { GenericEventMessage } from "./generic-event-message";
import { I18nKey } from "#/i18n/declaration";
import { useSlashCommandOutputStore } from "#/stores/slash-command-output-store";
import { getSkillDescription } from "#/utils/slash-command-description";

interface SlashCommandOutputMessagesProps {
  conversationId: string | null | undefined;
  anchorEventId: string | null;
}

export function SlashCommandOutputMessages({
  conversationId,
  anchorEventId,
}: SlashCommandOutputMessagesProps) {
  const { t } = useTranslation("openhands");
  const entriesByConversation = useSlashCommandOutputStore(
    (state) => state.entriesByConversation,
  );
  const entries = conversationId
    ? (entriesByConversation[conversationId] ?? []).filter(
        (entry) => entry.anchorEventId === anchorEventId,
      )
    : [];

  if (!conversationId || entries.length === 0) return null;

  return (
    <div
      data-testid="slash-command-output-messages"
      className="flex w-full flex-col"
    >
      {entries.map((entry) => {
        // @spec SC-002 — Inline help
        if (entry.kind === "help") {
          return (
            <GenericEventMessage
              key={entry.id}
              title={t(I18nKey.SLASH_COMMAND$HELP_TITLE, {
                count: entry.commands.length,
              })}
              details={
                <div className="mt-1 flex flex-col gap-2 text-sm text-neutral-300">
                  <div className="flex flex-col gap-1">
                    {entry.commands.map((item) => {
                      const description = item.descriptionKey
                        ? t(item.descriptionKey)
                        : (("description" in item.skill
                            ? item.skill.description
                            : null) ??
                          getSkillDescription(item.skill.content ?? ""));
                      return (
                        <div
                          key={item.command}
                          className="grid grid-cols-[minmax(7rem,auto)_1fr] gap-3 rounded-md border border-neutral-700 px-2 py-1.5"
                        >
                          <code className="text-neutral-100">
                            {item.command}
                          </code>
                          {description && <span>{description}</span>}
                        </div>
                      );
                    })}
                  </div>
                  <span>{t(I18nKey.SLASH_COMMAND$HELP_HINT)}</span>
                </div>
              }
              initiallyExpanded
            />
          );
        }

        return null;
      })}
    </div>
  );
}
