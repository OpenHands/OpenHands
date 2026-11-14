import React from "react";
import { useTranslation } from "react-i18next";
import { SmilePlus } from "lucide-react";
import type { ReactionEmoji } from "@openhands/typescript-client";
import { conversationReactionsService } from "#/api/conversation-reactions-service/conversation-reactions-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useIsExperimentalFeatureEnabled } from "#/hooks/query/use-experimental-features";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { useCloseOnEscape } from "#/hooks/use-close-on-escape";
import { I18nKey } from "#/i18n/declaration";
import { useEventStore, type OHEvent } from "#/stores/use-event-store";
import { cn } from "#/utils/utils";
import { displayApiErrorToast } from "#/utils/custom-toast-handlers";

export const CONVERSATION_REACTIONS_EXPERIMENTAL_FEATURE =
  "conversation_reactions";

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
type ReactionAuthor = "user" | "agent";

const compareEmoji = (a: string, b: string) =>
  EMOJI_ORDER.indexOf(a.split(":")[0] as ReactionEmoji) -
    EMOJI_ORDER.indexOf(b.split(":")[0] as ReactionEmoji) || a.localeCompare(b);

/**
 * The current reactions of every message in one events array, as a stable
 * string per message (`emoji:source` pairs in a fixed order, joined by
 * commas) so a message's selector only re-renders it when its own reactions
 * change. The latest event for a (message, emoji, source) wins. The index is
 * built once per events array, not once per message.
 */
const reactionIndexes = new WeakMap<object, Map<string, string>>();

function reactionIndex(events: readonly OHEvent[]): Map<string, string> {
  const cached = reactionIndexes.get(events);
  if (cached) return cached;

  const current = new Map<string, Map<string, boolean>>();
  for (const event of events) {
    if ("kind" in event && event.kind === "ReactionEvent") {
      const pairs = current.get(event.target_event_id) ?? new Map();
      pairs.set(`${event.emoji}:${event.source}`, !event.removed);
      current.set(event.target_event_id, pairs);
    }
  }

  const index = new Map<string, string>();
  for (const [target, pairs] of current) {
    const key = [...pairs.entries()]
      .filter(([, on]) => on)
      .map(([pair]) => pair)
      .sort(compareEmoji)
      .join(",");
    if (key) index.set(target, key);
  }
  reactionIndexes.set(events, index);
  return index;
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

export interface MessageReactionState {
  eventId: string;
  author: ReactionAuthor;
  key: string;
  canReact: boolean;
  conversationId: string | null | undefined;
  /** The message shows a reactions row: it has reactions or can get one. */
  visible: boolean;
}

/**
 * The reactions row state for one message. The message passes the row as
 * bubble children only when `visible`, so messages without one keep their
 * layout. The user can react only to the main agent's messages: a planning
 * agent's messages live in a sub-conversation this route does not address.
 */
export function useMessageReactions(
  eventId: string | undefined,
  author: ReactionAuthor,
  { isFromPlanningAgent = false }: { isFromPlanningAgent?: boolean } = {},
): MessageReactionState {
  const key = useEventStore((state) =>
    eventId ? (reactionIndex(state.events).get(eventId) ?? "") : "",
  );
  const enabled = useIsExperimentalFeatureEnabled(
    CONVERSATION_REACTIONS_EXPERIMENTAL_FEATURE,
  );
  const isCloud = useActiveBackend().backend.kind === "cloud";
  const { conversationId } = useOptionalConversationId();
  const canReact =
    !!eventId &&
    enabled &&
    !isCloud &&
    !!conversationId &&
    author === "agent" &&
    !isFromPlanningAgent;
  return {
    eventId: eventId ?? "",
    author,
    key,
    canReact,
    conversationId,
    visible: !!eventId && (key !== "" || canReact),
  };
}

const CLIPPING = new Set(["hidden", "clip", "auto", "scroll"]);

/** Edges, with an 8px margin, of the area that can show `element`: its
 * nearest ancestor that clips horizontally, else the window. */
function visibleBounds(element: HTMLElement): { left: number; right: number } {
  const margin = 8;
  for (let node = element.parentElement; node; node = node.parentElement) {
    if (CLIPPING.has(getComputedStyle(node).overflowX)) {
      const box = node.getBoundingClientRect();
      return { left: box.left + margin, right: box.right - margin };
    }
  }
  return { left: margin, right: window.innerWidth - margin };
}

/**
 * Emoji reactions under a chat message: the agent's (from its `react` tool)
 * and the user's. On the agent's messages the user can add or take off
 * their own reaction while the "Conversation reactions" feature is on.
 */
export function MessageReactions({
  state: { eventId, author, key, canReact, conversationId },
}: {
  state: MessageReactionState;
}) {
  const { t } = useTranslation("openhands");
  const [pickerOpen, setPickerOpen] = React.useState(false);
  const [pending, setPending] = React.useState(false);
  const triggerRef = React.useRef<HTMLButtonElement>(null);
  const closePicker = React.useCallback(() => setPickerOpen(false), []);
  const pickerRef = useClickOutsideElement<HTMLDivElement>(closePicker);
  const choicesRef = React.useRef<HTMLDivElement>(null);
  useCloseOnEscape(pickerOpen, closePicker, triggerRef);

  // Where the picker opens: beside its button, or above it, moved left by
  // `shift` px, when a narrow chat (a phone) would cut its last choices off.
  const [placement, setPlacement] = React.useState<{
    above: boolean;
    shift: number;
  }>({ above: false, shift: 0 });

  React.useLayoutEffect(() => {
    const choices = choicesRef.current;
    const anchor = pickerRef.current;
    if (!pickerOpen || !choices || !anchor) {
      setPlacement({ above: false, shift: 0 });
      return;
    }
    const bounds = visibleBounds(anchor);
    const picker = choices.getBoundingClientRect();
    if (picker.right <= bounds.right) return;
    const left = anchor.getBoundingClientRect().left;
    const fitted = Math.max(
      bounds.left,
      Math.min(left, bounds.right - picker.width),
    );
    setPlacement({ above: true, shift: fitted - left });
  }, [pickerOpen]);

  React.useEffect(() => {
    if (pickerOpen) {
      choicesRef.current?.querySelector<HTMLButtonElement>("button")?.focus();
    }
  }, [pickerOpen]);

  // The user's clicks show at once. The server's events replace them when
  // they arrive, which can be a while if a tool call is waiting for approval.
  const [optimistic, setOptimistic] = React.useState<
    Partial<Record<ReactionEmoji, boolean>>
  >({});
  React.useEffect(() => setOptimistic({}), [key]);

  const reactions = parseKey(key);
  for (const [emoji, on] of Object.entries(optimistic) as [
    ReactionEmoji,
    boolean,
  ][]) {
    const sources = reactions.get(emoji) ?? new Set<ReactionSource>();
    if (on) sources.add("user");
    else sources.delete("user");
    if (sources.size > 0) reactions.set(emoji, sources);
    else reactions.delete(emoji);
  }
  if (reactions.size === 0 && !canReact) return null;

  const toggle = async (emoji: ReactionEmoji) => {
    if (!canReact || !conversationId || pending) return;
    setPending(true);
    if (pickerOpen) {
      setPickerOpen(false);
      triggerRef.current?.focus();
    }
    const removed = reactions.get(emoji)?.has("user") ?? false;
    const before = optimistic;
    setOptimistic({ ...before, [emoji]: !removed });
    try {
      await conversationReactionsService.react({
        conversationId,
        emoji,
        targetEventId: eventId,
        removed,
      });
    } catch (error) {
      setOptimistic(before);
      displayApiErrorToast(error, t(I18nKey.CHAT_INTERFACE$REACTION_FAILED));
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
          aria-disabled={!canReact}
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
        <div ref={pickerRef} className="relative">
          <button
            ref={triggerRef}
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
              ref={choicesRef}
              role="group"
              aria-label={t(I18nKey.CHAT_INTERFACE$ADD_REACTION)}
              data-testid="reaction-picker"
              data-placement={placement.above ? "above" : "beside"}
              style={
                {
                  "--picker-shift": `${placement.shift}px`,
                } as React.CSSProperties
              }
              className={cn(
                "absolute z-10 flex gap-0.5 rounded-full border border-border bg-base-secondary px-1 py-0.5 shadow-sm",
                placement.above
                  ? // A measured shift, so it has no theme-scale value.
                    // eslint-disable-next-line shadcn/no-arbitrary-values
                    "bottom-full left-[var(--picker-shift)] mb-1"
                  : "left-7 top-1/2 -translate-y-1/2",
              )}
            >
              {EMOJI_ORDER.map((emoji) => (
                <button
                  key={emoji}
                  type="button"
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
