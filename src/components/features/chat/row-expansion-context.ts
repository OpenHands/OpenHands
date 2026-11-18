import React from "react";
import { messageExpansionKey } from "#/stores/message-expansion-store";

/**
 * The current rendered row's stable key (`renderedItemKey`), supplied by the
 * virtualized list. When present, the stateful controls inside the row lift
 * their expansion state into `useMessageExpansionStore` (keyed by row +
 * control) so virtualizing the row out of the DOM and back does not reset what
 * the user expanded.
 *
 * The value is a plain string, so re-rendering the list with the same rows does
 * not invalidate consumers the way a fresh callback would.
 *
 * `undefined` in the plain, non-virtualized list, where component state is
 * already stable.
 */
export const RowExpansionContext = React.createContext<string | undefined>(
  undefined,
);

/**
 * The event id of the card currently being rendered inside a row. A row can
 * hold several event cards at once (an expanded `EventGroup`, a thought plus
 * its action, model entries), so a control that appears once per card (a tool
 * output block, an error's details, a critic breakdown) needs the event id as
 * well as the row key to keep its expansion state to itself.
 *
 * `undefined` when the caller is not inside an event card.
 */
export const EventExpansionContext = React.createContext<string | undefined>(
  undefined,
);

/** Key for one control within the current row, or `undefined` outside a row. */
export function useRowExpansionKey(controlKey: string): string | undefined {
  const rowKey = React.useContext(RowExpansionContext);
  return React.useMemo(
    () =>
      rowKey === undefined
        ? undefined
        : messageExpansionKey(rowKey, controlKey),
    [rowKey, controlKey],
  );
}

/**
 * Key for one control within one event card of the current row, falling back to
 * {@link useRowExpansionKey} when no event card is in scope (so a control that
 * is unique per row — a user prompt, a thought — keeps working).
 */
export function useEventExpansionKey(controlKey: string): string | undefined {
  const rowKey = React.useContext(RowExpansionContext);
  const eventKey = React.useContext(EventExpansionContext);
  return React.useMemo(
    () =>
      rowKey === undefined
        ? undefined
        : messageExpansionKey(
            rowKey,
            eventKey === undefined ? controlKey : `${eventKey}::${controlKey}`,
          ),
    [rowKey, eventKey, controlKey],
  );
}
