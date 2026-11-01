import { useTranslation } from "react-i18next";
import { getDisplayConversationTags } from "#/api/agent-server-adapter";
import { cn } from "#/utils/utils";
import {
  formatConversationTagTooltip,
  truncateTagChipValue,
} from "../conversation-panel/conversation-card/conversation-tag-display";
import {
  getConversationTagIcon,
  type ConversationTagIcon,
} from "../conversation-panel/conversation-card/conversation-tag-icons";
import {
  CONVERSATION_CARD_META_CHIP_CLASSNAME,
  CONVERSATION_CARD_META_CHIP_ICON_CLASSNAME,
  CONVERSATION_CARD_META_CHIP_ICON_SLOT_CLASSNAME,
} from "../conversation-panel/conversation-card/conversation-card-meta-chip";

export interface ChannelOriginTag {
  key: string;
  value: string;
}

/**
 * The conversation's messaging-channel origin, if it carries one.
 *
 * Filters through {@link getDisplayConversationTags} so the reserved-key
 * exclusion and key normalization match the conversation-list chip exactly.
 * ``origin`` wins over ``source`` when both are stamped (``origin`` is the
 * canonical stamp); returns ``null`` when neither is present, so a
 * conversation without a channel tag renders no indicator.
 */
export function getChannelOriginTag(
  tags: Record<string, string> | null | undefined,
): ChannelOriginTag | null {
  const displayTags = getDisplayConversationTags(tags);
  const entry =
    displayTags.find(([key]) => key.trim().toLowerCase() === "origin") ??
    displayTags.find(([key]) => key.trim().toLowerCase() === "source");

  return entry ? { key: entry[0], value: entry[1] } : null;
}

/**
 * Receives the resolved icon as a prop (rather than assigning a local `Icon`
 * during render) so React sees a stable component type across renders.
 */
function ChannelOriginIcon({
  icon: Icon,
  keyName,
}: {
  icon: ConversationTagIcon;
  keyName: string;
}) {
  return (
    <span
      className={CONVERSATION_CARD_META_CHIP_ICON_SLOT_CLASSNAME}
      aria-hidden
    >
      <Icon
        data-testid="conversation-channel-origin-icon"
        data-tag-key={keyName}
        aria-hidden
        className={CONVERSATION_CARD_META_CHIP_ICON_CLASSNAME}
      />
    </span>
  );
}

interface ConversationChannelOriginIndicatorProps {
  /**
   * Server-side conversation tags (``AppConversation.tags``). Reserved keys are
   * filtered out, so passing the raw map is safe.
   */
  tags: Record<string, string> | null | undefined;
  className?: string;
}

/**
 * Read-only chip in the open-conversation header showing the messaging channel
 * a conversation came from (e.g. ``origin: slack``). It reuses the list chip's
 * label and icon helpers, so both surfaces render the same string and mark for
 * a given tag. Purely presentational: it creates, forks, or reroutes nothing,
 * and continuing the conversation keeps the same id.
 */
export function ConversationChannelOriginIndicator({
  tags,
  className,
}: ConversationChannelOriginIndicatorProps) {
  const { t } = useTranslation("openhands");
  const origin = getChannelOriginTag(tags);

  if (!origin) {
    return null;
  }

  const chipLabel = formatConversationTagTooltip(
    origin.key,
    truncateTagChipValue(origin.value),
    t,
  );

  return (
    <span
      data-testid="conversation-channel-origin"
      data-tag-key={origin.key}
      title={formatConversationTagTooltip(origin.key, origin.value, t)}
      aria-label={formatConversationTagTooltip(origin.key, origin.value, t)}
      className={cn(
        CONVERSATION_CARD_META_CHIP_CLASSNAME,
        "shrink min-w-0",
        className,
      )}
    >
      <ChannelOriginIcon
        icon={getConversationTagIcon(origin.key, origin.value)}
        keyName={origin.key}
      />
      <span className="truncate leading-4">{chipLabel}</span>
    </span>
  );
}
