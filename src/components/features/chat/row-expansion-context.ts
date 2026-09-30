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
