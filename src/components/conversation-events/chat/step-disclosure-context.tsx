import React from "react";
import { OpenHandsEvent } from "#/types/agent-server/core";
import {
  isACPToolCallEvent,
  isActionEvent,
  isObservationEvent,
  isUserRejectObservation,
} from "#/types/agent-server/type-guards";

/**
 * Open/closed state of one chat row, owned by the message list. A running
 * action is replaced by its observation, and a lone step can become part of a
 * group; both remount the row, so its state lives above it, keyed by tool call.
 * `open` is undefined until the user toggles the row.
 */
export interface StepDisclosure {
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
 */
export function useStepDisclosure(
  initiallyOpen: boolean,
): [boolean, () => void] {
  const shared = React.useContext(StepDisclosureContext);
  const [localOpen, setLocalOpen] = React.useState(initiallyOpen);
  if (!shared) return [localOpen, () => setLocalOpen((prev) => !prev)];
  const open = shared.open ?? initiallyOpen;
  return [open, () => shared.setOpen(!open)];
}
