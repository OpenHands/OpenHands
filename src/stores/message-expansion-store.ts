import React from "react";
import { create } from "zustand";

/**
 * Expansion state for the stateful controls inside conversation rows.
 *
 * Virtualization removes rows outside the viewport from the React tree, so any
 * `useState` living inside a row is destroyed when the row unmounts. Returning
 * to it would silently collapse an event group (or a tool detail) the user had
 * opened. Lifting those flags above the virtualized list — keyed by the row and
 * the nested control's identity — lets a remounted row restore the state the
 * user left it in.
 */
interface MessageExpansionState {
  expanded: Record<string, boolean>;
  setExpanded: (key: string, value: boolean) => void;
  /**
   * Drop entries whose row is no longer in `rowKeys` so a shrinking history
   * does not leak entries. Membership is decided on the row part of the key,
   * never on the whole key: a key also carries the nested control's identity
   * (see {@link messageExpansionKey}), so a raw set lookup would reject every
   * entry the moment any control key is present.
   */
  prune: (rowKeys: Set<string>) => void;
}

export const useMessageExpansionStore = create<MessageExpansionState>(
  (set) => ({
    expanded: {},
    setExpanded: (key, value) =>
      set((state) => ({ expanded: { ...state.expanded, [key]: value } })),
    prune: (rowKeys) =>
      set((state) => {
        const next: Record<string, boolean> = {};
        let changed = false;
        for (const [key, value] of Object.entries(state.expanded)) {
          if (rowKeys.has(expandedRowKey(key))) next[key] = value;
          else changed = true;
        }
        return changed ? { expanded: next } : state;
      }),
  }),
);

/**
 * Key for one control inside a rendered row. `rowKey` is a row's stable
 * `renderedItemKey`; `controlKey` distinguishes nested stateful controls
 * (e.g. one expanded tool card out of many in a group).
 */
export const messageExpansionKey = (rowKey: string, controlKey: string) =>
  `${rowKey}::${controlKey}`;

/** The row part of a {@link messageExpansionKey}, for pruning by row. */
export const expandedRowKey = (key: string): string => {
  const separator = key.indexOf("::");
  return separator === -1 ? key : key.slice(0, separator);
};

/**
 * Expansion flag that survives a row unmounting and remounting. With no `key`
 * (the plain, non-virtualized list) it degrades to ordinary component state, so
 * the plain list's behavior is unchanged.
 */
export function usePersistentExpansion(
  key: string | undefined,
  initiallyExpanded: boolean,
): readonly [boolean, () => void] {
  const [localExpanded, setLocalExpanded] = React.useState(initiallyExpanded);
  const stored = useMessageExpansionStore((state) =>
    key === undefined ? undefined : state.expanded[key],
  );
  const setStoredExpanded = useMessageExpansionStore(
    (state) => state.setExpanded,
  );

  const expanded =
    key === undefined ? localExpanded : (stored ?? initiallyExpanded);
  const toggle = React.useCallback(() => {
    if (key === undefined) {
      setLocalExpanded((value) => !value);
    } else {
      setStoredExpanded(key, !expanded);
    }
  }, [key, expanded, setStoredExpanded]);

  return [expanded, toggle] as const;
}
