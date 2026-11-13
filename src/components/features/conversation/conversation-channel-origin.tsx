import { useTranslation } from "react-i18next";
import { getDisplayConversationTags } from "#/api/agent-server-adapter";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { I18nKey } from "#/i18n/declaration";
import { ConversationTagChip } from "../conversation-panel/conversation-card/conversation-tag-chips";

/** Tag keys that name the channel a conversation came from (Slack, an API). */
const CHANNEL_ORIGIN_TAG_KEYS: ReadonlySet<string> = new Set([
  "origin",
  "source",
]);

/**
 * Read-only chip in the chat header for a conversation that carries an
 * ``origin`` / ``source`` tag (for example ``origin: slack``). It shows the
 * same chip as the conversation list, so the user can see where the
 * conversation came from after they open it. Renders nothing otherwise.
 */
export function ConversationChannelOrigin() {
  const { t } = useTranslation("openhands");
  const { data: conversation } = useActiveConversation();

  // Bare tags name no channel, so they get no chip here.
  const originTags = getDisplayConversationTags(conversation?.tags).filter(
    ([key, value]) =>
      CHANNEL_ORIGIN_TAG_KEYS.has(key.trim().toLowerCase()) &&
      value.trim().length > 0,
  );

  if (originTags.length === 0) {
    return null;
  }

  return (
    <div
      data-testid="conversation-channel-origin"
      className="ml-1 flex shrink-0 items-center gap-1"
    >
      {originTags.map(([key, value]) => (
        <ConversationTagChip
          key={key}
          tagKey={key}
          value={value}
          title={t(I18nKey.CONVERSATION$CHANNEL_ORIGIN_TOOLTIP, {
            origin: value,
          })}
          testId="conversation-channel-origin-chip"
          iconTestId="conversation-channel-origin-icon"
          // In a narrow header (phone, or a chat pane beside the right panel)
          // the title keeps the room and the chip shows only its icon. The
          // label stays readable for screen readers.
          labelClassName="sr-only @md:not-sr-only"
        />
      ))}
    </div>
  );
}
