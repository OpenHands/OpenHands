import { getLastRenderableEventId } from "#/hooks/chat/model-command-event-anchor";
import {
  useModelStore,
  SeededSwitch,
  ModelSwitchDetails,
} from "#/stores/model-store";
import {
  getStoredConversationMetadata,
  mergeStoredConversationMetadata,
  type ConversationMetadata,
} from "#/api/conversation-metadata-store";
import { OpenHandsEvent } from "#/types/agent-server/core";
import {
  isClassifyAndSwitchLLMObservationEvent,
  isSwitchLLMObservationEvent,
} from "#/types/agent-server/type-guards";
import { shouldRenderEvent } from "#/components/conversation-events/chat/event-content-helpers/should-render-event";
import { useEventStore } from "#/stores/use-event-store";

export function recordModelSwitchMessage(
  conversationId: string,
  profileName: string,
  anchorEventId: string | null = getLastRenderableEventId(),
  details?: ModelSwitchDetails,
) {
  useModelStore
    .getState()
    .recordSwitch(conversationId, anchorEventId, profileName, details);
}

/**
 * The active-profile stamp for a conversation: the optimistic in-memory entry
 * (read first by the chat-input pill) plus the persisted per-conversation
 * `active_profile` metadata (survives reloads, round-tripped onto the
 * conversation by the agent-server adapter — issue #1082). Every path that
 * learns of a successful switch — the user `/model` mutation, the live
 * WebSocket handler, and the history seed below — must write through this one
 * function so the live and reload paths can't drift.
 *
 * `stampedAfterEventId` is persisted alongside the profile so the history seed
 * can tell a stale observation apart from a newer manual switch (see below).
 * The live WebSocket handler passes the observation's own id; the user
 * `/model` mutation has no event and defaults to the newest event this
 * conversation has loaded.
 */
export function stampActiveLlmProfile(
  conversationId: string,
  profileName: string,
  stampedAfterEventId: string | null = getNewestEventId(conversationId),
) {
  useModelStore.getState().setActiveProfile(conversationId, profileName);

  // Merge rather than replace: the stamp only owns these two fields, and the
  // record carries others it must not drop — the attached workspace mode
  // (#15520), the plugins snapshot behind the in-conversation plugins view,
  // and the local planner conversation id. Enumerating them here meant every
  // new field had to remember to opt in.
  mergeStoredConversationMetadata(conversationId, {
    active_profile: profileName,
    stamped_after_event_id: stampedAfterEventId,
  });
}

/**
 * The newest persisted main-conversation event loaded for `conversationId`,
 * if any. Transient events are skipped: they are never in REST history, so a
 * reload could not find them.
 */
function getNewestEventId(conversationId: string): string | null {
  const { events, loadedConversationId } = useEventStore.getState();
  if (loadedConversationId !== conversationId) return null;
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (!event.isFromPlanningAgent && !event.isTransient && event.id) {
      return String(event.id);
    }
  }
  return null;
}

/**
 * Whether the latest switch in history is newer than the stored stamp.
 *
 * Compared by position in the event log rather than by time: the agent-server
 * writes offset-less timestamps in its own local time zone, so comparing them
 * against the client clock is off by the server's UTC offset (hours in either
 * direction, depending on where the agent-server runs). Loaded history is
 * always the newest stretch of the log, so a stamp anchored on an event that
 * is not loaded predates every loaded switch.
 */
function isNewerThanStamp(
  switchEventId: string,
  switchTimestamp: string,
  stored: ConversationMetadata | null,
  events: OpenHandsEvent[],
): boolean {
  if (!stored?.active_profile) return true;

  const anchorId = stored.stamped_after_event_id;
  if (anchorId !== undefined) {
    const ids = events.map((event) => String(event.id));
    const anchorIndex = anchorId === null ? -1 : ids.indexOf(anchorId);
    return anchorIndex < ids.indexOf(switchEventId);
  }

  // Stamps written before `stamped_after_event_id` existed only carry a time.
  const stampedAt = stored.stamped_at;
  return !stampedAt || Date.parse(switchTimestamp) > Date.parse(stampedAt);
}

/**
 * Rebuilds the inline "Switched to" messages for a conversation from its loaded
 * history.
 *
 * The live messages live in an in-memory store written only by the WebSocket
 * handler and the user `/model` action. Existing conversations load via REST
 * history, which bypasses that handler, so without this replay no past agent
 * switches would render after a reload (the SwitchLLMObservation events are
 * also hidden as cards by `shouldRenderEvent`).
 *
 * `uiEvents` MUST be the event store's `uiEvents` (the same list the renderer
 * and the live `getLastRenderableEventId()` use) — NOT the raw history. The
 * renderer anchors a message after an event only if that event's id is in
 * `uiEvents.filter(shouldRenderEvent)`, and `uiEvents` differs from raw history:
 * actions are replaced by their observations and `ThinkObservation` /
 * `FinishObservation` are dropped. Anchoring off raw history would point at ids
 * that never mount (e.g. a dropped `ThinkObservation`), orphaning the message.
 *
 * Each successful switch is anchored to the last renderable event before it,
 * matching where the live handler would have placed it. Idempotent: entries are
 * keyed by the observation event id, so re-seeding on every reload is a no-op.
 *
 * Also re-derives the active-profile stamp from the latest successful
 * SwitchLLM observation. The stamp is otherwise written only by the live
 * WebSocket handler and the user `/model` mutation, so a switch that fired
 * while the socket was down would never be stamped — after a reload the pill
 * showed the stale profile while the panel showed the server's true model.
 *
 * The re-stamp is gated on the stored stamp: manual `/model` switches go
 * through REST and leave NO observation in history, so an unconditional
 * re-stamp would roll a newer manual switch back to an older tool switch.
 * The seed therefore only stamps when there is no existing stamp or the
 * latest observation comes after the stamp's anchor event in `events` (see
 * `isNewerThanStamp`). `events` is the store's raw, unprojected list: the
 * anchor can be an event `uiEvents` replaces or drops.
 */
export function seedModelSwitchesFromHistory(
  conversationId: string,
  uiEvents: OpenHandsEvent[],
  events: OpenHandsEvent[] = uiEvents,
) {
  const switches: SeededSwitch[] = [];
  let lastRenderableId: string | null = null;
  let latestSwitch: {
    eventId: string;
    profileName: string;
    timestamp: string;
  } | null = null;

  for (const event of uiEvents) {
    if (isSwitchLLMObservationEvent(event) && !event.observation.is_error) {
      switches.push({
        id: `history-switch:${event.id}`,
        anchorEventId: lastRenderableId,
        profileName: event.observation.profile_name,
        initiator: "agent",
        reason: event.observation.reason,
      });
      latestSwitch = {
        eventId: String(event.id),
        profileName: event.observation.profile_name,
        timestamp: event.timestamp,
      };
    } else if (
      isClassifyAndSwitchLLMObservationEvent(event) &&
      !event.observation.is_error &&
      event.observation.model
    ) {
      // Router-driven switch: the matched LLM profile name is the
      // observation's `model` (the SDK matched classifier output against
      // saved profile names case-insensitively and stamped the profile name
      // there; `active_model` is the underlying model string, not the
      // profile name).
      const profileName = event.observation.model;
      switches.push({
        id: `history-classify-switch:${event.id}`,
        anchorEventId: lastRenderableId,
        profileName,
        initiator: "router",
        reason: event.observation.chosen_class,
      });
      latestSwitch = {
        eventId: String(event.id),
        profileName,
        timestamp: event.timestamp,
      };
    }
    if (shouldRenderEvent(event)) {
      lastRenderableId = String(event.id);
    }
  }

  if (switches.length > 0) {
    useModelStore.getState().seedSwitches(conversationId, switches);
  }

  if (latestSwitch) {
    const { eventId, profileName, timestamp } = latestSwitch;
    const stored = getStoredConversationMetadata(conversationId);
    if (isNewerThanStamp(eventId, timestamp, stored, events)) {
      // Anchor on the observation itself so re-seeding the same history on
      // the next reload is a no-op (strictly-newer never matches).
      stampActiveLlmProfile(conversationId, profileName, eventId);
    }
  }
}
