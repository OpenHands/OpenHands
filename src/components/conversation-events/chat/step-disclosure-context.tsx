import React from "react";
import { OpenHandsEvent } from "#/types/agent-server/core";
import {
  isACPToolCallEvent,
  isActionEvent,
  isObservationEvent,
  isUserRejectObservation,
} from "#/types/agent-server/type-guards";
import { usePersistentExpansion } from "#/stores/message-expansion-store";

/**
 * Open/closed state of one chat row, owned by the message list. A running
 * action is replaced by its observation, and a lone step can become part of a
 * group; both remount the row, so its state lives above it, keyed by tool call.
 * `open` is undefined until the user toggles the row.
 */
export interface StepDisclosure {
  detailsOnly?: boolean;
  open: boolean | undefined;
  setOpen: (open: boolean) => void;
}

export const StepDisclosureContext = React.createContext<StepDisclosure | null>(
  null,
);

/**
 * Stable key for a step. An action and the observation or rejection that
 * replaces it share their tool call; other events keep their own id (an agent
 * error renders beside its action, so it must not share the action's key).
 */
export const getStepKey = (event: OpenHandsEvent): string =>
  isActionEvent(event) ||
  isObservationEvent(event) ||
  isUserRejectObservation(event) ||
  isACPToolCallEvent(event)
    ? `call:${event.tool_call_id}`
    : String(event.id);

/**
 * The row's open state: shared through the list when one is provided,
 * otherwise local to the component.
 *
 * `persistedKey` names the fallback in `useMessageExpansionStore` for a control
 * that is not part of a step (a /model listing, whose disclosure would
 * otherwise be destroyed when virtualization unmounts its row). It only applies
 * when no shared step state is in scope; without it the fallback stays
 * component-local, as it is in the plain, non-virtualized list.
 */
export function useStepDisclosure(
  initiallyOpen: boolean,
  persistedKey?: string,
): [boolean, () => void] {
  const shared = React.useContext(StepDisclosureContext);
  const [persistedOpen, togglePersisted] = usePersistentExpansion(
    shared ? undefined : persistedKey,
    initiallyOpen,
  );
  if (!shared) return [persistedOpen, togglePersisted];
  const open = shared.open ?? initiallyOpen;
  return [open, () => shared.setOpen(!open)];
}
