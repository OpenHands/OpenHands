import { getLastRenderableEventId } from "#/hooks/chat/model-command-event-anchor";
import { useModelStore, SeededSwitch } from "#/stores/model-store";
import { OpenHandsEvent } from "#/types/agent-server/core";
import { isSwitchLLMObservationEvent } from "#/types/agent-server/type-guards";
import { shouldRenderEvent } from "#/components/conversation-events/chat/event-content-helpers/should-render-event";

export function recordModelSwitchMessage(
  conversationId: string,
  profileName: string,
  anchorEventId: string | null = getLastRenderableEventId(),
) {
  useModelStore
    .getState()
    .recordSwitch(conversationId, anchorEventId, profileName);
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
 */
export function seedModelSwitchesFromHistory(
  conversationId: string,
  uiEvents: OpenHandsEvent[],
) {
  const switches: SeededSwitch[] = [];
  let lastRenderableId: string | null = null;

  for (const event of uiEvents) {
    if (isSwitchLLMObservationEvent(event) && !event.observation.is_error) {
      switches.push({
        id: `history-switch:${event.id}`,
        anchorEventId: lastRenderableId,
        profileName: event.observation.profile_name,
      });
    }
    if (shouldRenderEvent(event)) {
      lastRenderableId = String(event.id);
    }
  }

  if (switches.length > 0) {
    useModelStore.getState().seedSwitches(conversationId, switches);
  }
}
