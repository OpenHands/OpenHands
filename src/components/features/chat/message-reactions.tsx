import React from "react";
import { useTranslation } from "react-i18next";
import { SmilePlus } from "lucide-react";
import type { ReactionEmoji } from "@openhands/typescript-client";
import { featuresService } from "#/api/features-service/features-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useIsFeatureEnabled } from "#/hooks/query/use-features";
import { I18nKey } from "#/i18n/declaration";
import { useEventStore } from "#/stores/use-event-store";
import { cn } from "#/utils/utils";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

export const CONVERSATION_REACTIONS_FEATURE = "conversation_reactions";

const EMOJI_GLYPHS: Record<ReactionEmoji, string> = {
  eyes: "👀",
  "+1": "👍",
  white_check_mark: "✅",
  tada: "🎉",
  thinking_face: "🤔",
  heart: "❤️",
};
const EMOJI_ORDER = Object.keys(EMOJI_GLYPHS) as ReactionEmoji[];
// Theme fonts (Paper's serif, for one) can carry plain glyphs for some emoji;
// prefer the platform's colour emoji font.
const EMOJI_FONT = "font-emoji";

type ReactionSource = "agent" | "user";

interface StoredReaction {
  kind?: string;
  target_event_id?: string;
  emoji?: string;
  removed?: boolean;
  source?: string;
}

/**
 * The current reactions on one message, as a stable string so the store
 * selector only re-renders this row when they change: `emoji:source` pairs in
 * a fixed order, joined by commas. The latest event for an (emoji, source)
 * pair wins.
 */
export function reactionsKey(events: unknown[], targetEventId: string): string {
  const current = new Map<string, boolean>();
  for (const raw of events) {
    const event = raw as StoredReaction;
    if (
      event.kind === "ReactionEvent" &&
      event.target_event_id === targetEventId &&
      event.emoji &&
      (event.source === "agent" || event.source === "user")
    ) {
      current.set(`${event.emoji}:${event.source}`, !event.removed);
    }
  }
  return [...current.entries()]
    .filter(([, on]) => on)
    .map(([pair]) => pair)
    .sort(
      (a, b) =>
        EMOJI_ORDER.indexOf(a.split(":")[0] as ReactionEmoji) -
          EMOJI_ORDER.indexOf(b.split(":")[0] as ReactionEmoji) ||
        a.localeCompare(b),
    )
    .join(",");
}

function parseKey(key: string): Map<ReactionEmoji, Set<ReactionSource>> {
  const byEmoji = new Map<ReactionEmoji, Set<ReactionSource>>();
  if (!key) return byEmoji;
  for (const pair of key.split(",")) {
    const [emoji, source] = pair.split(":") as [ReactionEmoji, ReactionSource];
    if (!(emoji in EMOJI_GLYPHS)) continue;
    const sources = byEmoji.get(emoji) ?? new Set<ReactionSource>();
    sources.add(source);
    byEmoji.set(emoji, sources);
  }
  return byEmoji;
}

interface MessageReactionsProps {
  /** The message (or finish action) the reactions are on. */
  eventId: string;
  /** Who wrote the message: chips line up with its bubble. */
  author: "user" | "agent";
}

function useReactionState(eventId: string, author: "user" | "agent") {
  const key = useEventStore((state) => reactionsKey(state.events, eventId));
  const enabled = useIsFeatureEnabled(CONVERSATION_REACTIONS_FEATURE);
  const isCloud = useActiveBackend().backend.kind === "cloud";
  const { conversationId } = useOptionalConversationId();
  const canReact =
    enabled && !isCloud && !!conversationId && author === "agent";
  return { key, canReact, conversationId };
}

/**
 * Whether a message shows a reactions row: it has reactions, or the person
 * can add one. The message passes the row as bubble children only then, so
 * messages without one keep their layout.
 */
export function useHasMessageReactions(
  eventId: string | undefined,
  author: "user" | "agent",
): boolean {
  const { key, canReact } = useReactionState(eventId ?? "", author);
  return !!eventId && (key !== "" || canReact);
}

/**
 * Emoji reactions under a chat message: the agent's (from its `react` tool)
 * and the person's. On the agent's messages the person can add or take off
 * their own reaction while the "Conversation reactions" feature is on.
 */
export function MessageReactions({ eventId, author }: MessageReactionsProps) {
  const { t } = useTranslation("openhands");
  const { key, canReact, conversationId } = useReactionState(eventId, author);
  const [pickerOpen, setPickerOpen] = React.useState(false);
  const [pending, setPending] = React.useState(false);

  const reactions = parseKey(key);
  if (reactions.size === 0 && !canReact) return null;

  const toggle = async (emoji: ReactionEmoji) => {
    if (!canReact || !conversationId || pending) return;
    setPending(true);
    setPickerOpen(false);
    try {
      await featuresService.react({
        conversationId,
        emoji,
        targetEventId: eventId,
        removed: reactions.get(emoji)?.has("user") ?? false,
      });
    } catch (error) {
      displayErrorToast(error instanceof Error ? error.message : null);
    } finally {
      setPending(false);
    }
  };

  const describe = (sources: Set<ReactionSource>) =>
    [
      sources.has("agent") ? t(I18nKey.CHAT_INTERFACE$REACTED_BY_AGENT) : null,
      sources.has("user") ? t(I18nKey.CHAT_INTERFACE$REACTED_BY_YOU) : null,
    ]
      .filter(Boolean)
      .join(" · ");

  return (
    <div
      data-testid="message-reactions"
      className={cn(
        "flex flex-wrap items-center gap-1",
        author === "user" && "self-end",
      )}
    >
      {[...reactions.entries()].map(([emoji, sources]) => (
        <button
          key={emoji}
          type="button"
          data-testid={`reaction-${emoji}`}
          data-sources={[...sources].sort().join(",")}
          title={describe(sources)}
          aria-label={`${EMOJI_GLYPHS[emoji]} ${describe(sources)}`}
          aria-pressed={sources.has("user")}
          disabled={!canReact}
          onClick={() => toggle(emoji)}
          className={cn(
            "flex h-6 items-center gap-1 rounded-full border border-border px-2 text-xs leading-none",
            sources.has("user") ? "bg-interactive-hover" : "bg-transparent",
            canReact
              ? "cursor-pointer hover:bg-interactive-hover"
              : "cursor-default",
          )}
        >
          <span aria-hidden className={cn("text-base", EMOJI_FONT)}>
            {EMOJI_GLYPHS[emoji]}
          </span>
          {sources.size > 1 && (
            <span className="text-muted">{sources.size}</span>
          )}
        </button>
      ))}
      {canReact && (
        <div className="relative">
          <button
            type="button"
            data-testid="add-reaction"
            title={t(I18nKey.CHAT_INTERFACE$ADD_REACTION)}
            aria-label={t(I18nKey.CHAT_INTERFACE$ADD_REACTION)}
            aria-expanded={pickerOpen}
            disabled={pending}
            onClick={() => setPickerOpen((open) => !open)}
            className="flex h-6 w-6 cursor-pointer items-center justify-center rounded-full text-muted hover:bg-interactive-hover hover:text-foreground"
          >
            <SmilePlus className="size-3.5" strokeWidth={2} aria-hidden />
          </button>
          {pickerOpen && (
            <div
              role="menu"
              data-testid="reaction-picker"
              tabIndex={-1}
              onKeyDown={(event) => {
                if (event.key === "Escape") setPickerOpen(false);
              }}
              className="absolute left-7 top-1/2 z-10 -translate-y-1/2 flex gap-0.5 rounded-full border border-border bg-base-secondary px-1 py-0.5 shadow-sm"
            >
              {EMOJI_ORDER.map((emoji) => (
                <button
                  key={emoji}
                  type="button"
                  role="menuitem"
                  data-testid={`reaction-picker-${emoji}`}
                  aria-label={EMOJI_GLYPHS[emoji]}
                  onClick={() => toggle(emoji)}
                  className={cn(
                    "flex h-6 w-6 cursor-pointer items-center justify-center rounded-full text-sm hover:bg-interactive-hover",
                    EMOJI_FONT,
                  )}
                >
                  {EMOJI_GLYPHS[emoji]}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
