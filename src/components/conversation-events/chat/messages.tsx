import React from "react";
import { useVirtualizer, type VirtualItem } from "@tanstack/react-virtual";
import { OpenHandsEvent } from "#/types/agent-server/core";
import {
  isActionEvent,
  isObservationEvent,
  isPlanningFileEditorObservationEvent,
  isUserRejectObservation,
} from "#/types/agent-server/type-guards";
import { cn } from "#/utils/utils";
import { EventMessage } from "./event-message";
import { usePlanPreviewEvents } from "./hooks/use-plan-preview-events";
import { groupEvents, renderedItemKey, RenderedItem } from "./group-events";
import { buildActionById } from "./event-thought-helpers";
import { EventGroup } from "./event-message-components/event-group";
import { ThoughtEventMessage } from "./event-message-components/thought-event-message";
import {
  PLAIN_ROW_MARGIN_RESET_CLASSNAME,
  VIRTUAL_ROW_MARGIN_RESET_CLASSNAME,
} from "./row-margins";
import { useModelStore } from "#/stores/model-store";
import { ModelMessages } from "#/components/features/chat/model-messages";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { ConversationConfirmationButtons } from "#/components/shared/buttons/conversation-confirmation-buttons";
import {
  EventExpansionContext,
  RowExpansionContext,
} from "#/components/features/chat/row-expansion-context";
import { useMessageExpansionStore } from "#/stores/message-expansion-store";
import {
  buildInitialMeasurements,
  measurePlainRow,
  anchorScrollOnItemSizeChange,
} from "./scroll-anchoring";
import type { SecretRequest } from "#/stores/secret-request-store";
import {
  SecretRequestCard,
  usePendingSecretRequests,
} from "#/components/features/conversation/secret-request-card";

interface MessagesProps {
  messages: OpenHandsEvent[]; // UI events (actions replaced by observations)
  allEvents: OpenHandsEvent[]; // Full event history (for action lookup)
  /**
   * The chat scroll container element. When provided and the conversation is
   * long enough, rows are virtualized against it so only a viewport's worth of
   * message DOM is mounted. Omit it (shared/read-only views, isolated mounts)
   * to always render the plain list.
   *
   * This is the *element*, not a ref: the virtualizer only attaches once the
   * element exists, and a ref owned by an ancestor never re-renders this
   * component when it is populated. `undefined` means "no virtualization
   * context"; `null` means "context exists, element not mounted yet" — the
   * virtualized shell renders (empty for one frame) instead of falling back to
   * the full plain list, so a long conversation never pays for a full render.
   */
  scrollParent?: HTMLDivElement | null;
}

/**
 * Row count at which the list switches from plain rendering to virtualization.
 * Below this, rendering every row is cheaper than the virtualizer's
 * measurement churn, and short conversations stay byte-identical.
 *
 * 150, not the ~50 first sketched in #18102: a rendered row is a *grouped* row,
 * not one event, so 150 rows is already a multi-hundred-event conversation. At
 * that size the plain list is still only ~1.6k DOM nodes, so it stays cheap
 * enough that the virtualizer's measurement churn is not worth paying for
 * earlier; above it the mounted window is bounded regardless of history.
 */
const VIRTUALIZATION_THRESHOLD = 150;
/** Rows to keep mounted beyond the viewport on each side. */
const OVERSCAN = 8;
/** Pre-measurement row height guess, corrected once rows mount. */
const ESTIMATED_ROW_HEIGHT = 120;
/**
 * Vertical gap between rows, matching the plain list's `gap-2`. Absolutely
 * positioned virtual rows drop out of flex flow, so the gap has to be part of
 * each measured row instead of the container.
 */
const ROW_GAP_PX = 8;

const getLastEventId = (events: OpenHandsEvent[]) => events.at(-1)?.id;
const getLastEvent = (events: OpenHandsEvent[]) => events.at(-1);

export const Messages: React.FC<MessagesProps> = React.memo(
  ({ messages, allEvents, scrollParent }) => {
    const { conversationId } = useOptionalConversationId();
    // Get the set of event IDs that should render PlanPreview
    // This ensures only one preview per user message "phase"
    const planPreviewEventIds = usePlanPreviewEvents(allEvents);

    // EventMessage used to receive the complete allEvents array and the plan
    // Set, so every append changed every historical item's shallow props.
    // Derive the two event-specific values once and keep unchanged wrappers
    // eligible for React.memo without removing allEvents from grouping logic.
    const actionById = React.useMemo(
      () => buildActionById(allEvents),
      [allEvents],
    );

    // Set of event ids that have a /model entry anchored to them — used to
    // avoid mounting <ModelMessages> for every event (the component would
    // otherwise early-return null).
    const modelEntries = useModelStore((s) =>
      conversationId ? s.entriesByConversation[conversationId] : undefined,
    );
    const modelAnchorIds = React.useMemo(() => {
      if (!modelEntries || modelEntries.length === 0) return null;
      const ids = new Set<string>();
      for (const entry of modelEntries) {
        if (entry.anchorEventId !== null) ids.add(entry.anchorEventId);
      }
      return ids.size > 0 ? ids : null;
    }, [modelEntries]);

    const maybeRenderModelMessages = (eventId: string | number | undefined) => {
      if (!modelAnchorIds || eventId === undefined) return null;
      const key = String(eventId);
      if (!modelAnchorIds.has(key)) return null;
      return (
        <ModelMessages conversationId={conversationId} anchorEventId={key} />
      );
    };

    // Pending `request_secret` forms sit right after their tool call, so what
    // the agent writes next lands below them. A call that is not loaded falls
    // back to the end of the chat.
    const pendingSecretRequests = usePendingSecretRequests();
    const secretRequests = React.useMemo(() => {
      const unanchored = new Map(
        pendingSecretRequests.map((request) => [request.toolCallId, request]),
      );
      const byAnchorId = new Map<string, SecretRequest>();
      for (const event of messages) {
        if (unanchored.size === 0) break;
        if (!isActionEvent(event) && !isObservationEvent(event)) continue;
        const request = unanchored.get(event.tool_call_id);
        if (!request) continue;
        unanchored.delete(event.tool_call_id);
        byAnchorId.set(event.id, request);
      }
      return { byAnchorId, unanchored: [...unanchored.values()] };
    }, [pendingSecretRequests, messages]);

    const maybeRenderSecretRequest = (eventId: string | undefined) => {
      const request = eventId && secretRequests.byAnchorId.get(eventId);
      return request ? <SecretRequestCard request={request} /> : null;
    };

    // Fold consecutive action/observation events into collapsible groups so a
    // long sequence of tool calls doesn't dominate the chat scroll. Items that
    // can't be grouped (or that fall in a short run) are still rendered one by
    // one, identically to before. Agent thoughts attached to an action are
    // hoisted out as their own rendered item so they always show up in the
    // message pane and a thought between actions starts a fresh group.
    const plainItems = React.useMemo(
      () => groupEvents(messages, undefined, allEvents, actionById),
      [messages, allEvents, actionById],
    );

    // `null` (not `undefined`) means the caller has a scroll container but the
    // element is not mounted yet: virtualize the shell rather than rendering
    // every row for a frame.
    const shouldVirtualize = Boolean(
      scrollParent !== undefined &&
      plainItems.length >= VIRTUALIZATION_THRESHOLD,
    );

    // A planning observation that is not the designated preview for its phase
    // renders nothing. In the plain list that is invisible (a null render
    // contributes no height), but a virtualized row would reserve an estimate
    // for it and leave blank scroll space, so drop those rows on the
    // virtualized path only. The plain list keeps them, so the EventMessage
    // props (and therefore `showPlanPreview`) stay observable for every event.
    //
    // A row can also carry a /model entry, and `getLastRenderableEventId`
    // anchors those to a planning observation. Dropping such a row would take
    // the entry with it, so rows that anchor one are kept: their event content
    // still renders null (exactly as in the plain list) while the entry shows.
    const renderedItems = React.useMemo(
      () =>
        shouldVirtualize
          ? plainItems.filter(
              (item) =>
                item.kind !== "single" ||
                !isPlanningFileEditorObservationEvent(item.event) ||
                (!!item.event.id && planPreviewEventIds.has(item.event.id)) ||
                (!!item.event.id && modelAnchorIds?.has(item.event.id)),
            )
          : plainItems,
      [plainItems, shouldVirtualize, planPreviewEventIds, modelAnchorIds],
    );

    // Whether a group is still the live tail is decided from `plainItems` in
    // both modes. Filtering a non-preview planning observation (above) makes the
    // virtualized row set shorter, so reading the tail off `renderedItems` made
    // the two modes disagree about the last group: a group followed only by a
    // hidden planning observation is finalized in the plain list but looked live
    // in the virtualized one. The hidden row renders nothing, so it cannot be
    // "after" the group.
    const tailItemKey = plainItems.length
      ? renderedItemKey(plainItems[plainItems.length - 1])
      : undefined;

    const renderEventMessage = (
      event: OpenHandsEvent,
      index: number,
      suppressThought: boolean,
    ) => (
      // Scope the event id for the card's own expansion controls, so several
      // cards sharing one virtualized row (an event group, a thought plus its
      // action) don't collapse and expand together.
      <EventExpansionContext.Provider
        key={event.id}
        value={String(event.id ?? "")}
      >
        <EventMessage
          event={event}
          correspondingAction={
            (isObservationEvent(event) || isUserRejectObservation(event)) &&
            event.action_id
              ? (actionById.get(event.action_id) ?? null)
              : null
          }
          isLastMessage={messages.length - 1 === index}
          isInLast10Actions={messages.length - 1 - index < 10}
          showPlanPreview={event.id ? planPreviewEventIds.has(event.id) : false}
          suppressThought={suppressThought}
        />
      </EventExpansionContext.Provider>
    );

    const renderItem = (item: RenderedItem) => {
      if (item.kind === "single") {
        return (
          <>
            {/* Thoughts for singles are also hoisted as their own
                "thought" item, so suppress the inline render to avoid
                duplication. */}
            {renderEventMessage(item.event, item.index, true)}
            {maybeRenderModelMessages(item.event.id)}
            {maybeRenderSecretRequest(item.event.id)}
          </>
        );
      }

      if (item.kind === "thought") {
        return (
          <>
            <ThoughtEventMessage event={item.action} />
            {maybeRenderModelMessages(item.action.id)}
          </>
        );
      }

      // A group is "finalized" once another rendered item appears after
      // it, signalling the agent has moved on. While the group is still
      // the live tail, it keeps showing the latest action title as its
      // prominent summary.
      const isFinalized = renderedItemKey(item) !== tailItemKey;
      return (
        <>
          <EventGroup
            events={item.events}
            allEvents={allEvents}
            isFinalized={isFinalized}
          >
            {item.events.map((event, offset) =>
              renderEventMessage(event, item.startIndex + offset, true),
            )}
          </EventGroup>
          {item.events.map((event) => (
            <React.Fragment key={`model-${event.id}`}>
              {maybeRenderModelMessages(event.id)}
              {maybeRenderSecretRequest(event.id)}
            </React.Fragment>
          ))}
        </>
      );
    };

    // A collapsed history (or a switch to another conversation) must not keep
    // stale expansion entries for rows that no longer exist.
    const rowKeys = React.useMemo(
      () => renderedItems.map(renderedItemKey),
      [renderedItems],
    );
    const pruneExpansion = useMessageExpansionStore((s) => s.prune);
    React.useEffect(() => {
      pruneExpansion(new Set(rowKeys));
    }, [rowKeys, pruneExpansion]);

    // Stable item-key accessor. TanStack compares this callback by identity when
    // deciding whether to rebuild its measurement cache; an inline arrow would
    // look like a new key extractor on every render and invalidate the cache
    // (recomputing offsets and remounting rows for no reason).
    const getItemKey = React.useCallback(
      (index: number) => rowKeys[index],
      [rowKeys],
    );

    // The virtualizer's scroll element is the chat column, which also holds
    // content *before* this list (top-anchored /model cards, the older-history
    // spinner). TanStack compares the container's `scrollTop` against item
    // starts, so without the list's own offset it believes the list starts at
    // 0 and mounts the wrong rows — leaving blank space near the top once the
    // preceding content is taller than the overscan. Measure the shell's
    // offset within the scroll element and pass it as `scrollMargin`.
    const [scrollMargin, setScrollMargin] = React.useState(0);

    // Per-item heights of the plain list, keyed by row key. Captured while the
    // plain list is mounted so that crossing the virtualization threshold can
    // seed the virtualizer with the real geometry (see
    // `buildInitialMeasurements`): seeding the shell with the plain list's own
    // heights keeps its total height — and therefore the reader's scroll
    // offset — unchanged across the switch, instead of collapsing to a flat
    // estimate and jumping.
    const plainListRef = React.useRef<HTMLDivElement | null>(null);
    const plainItemHeights = React.useRef<Map<string, number>>(new Map());
    // One observer for the whole plain phase. Rebuilding it on every append
    // would re-observe every row, and each `observe` delivers an initial
    // callback that re-measures it — so the work would grow with the list on
    // every append instead of tracking just the rows that changed.
    const plainRowObserver = React.useRef<ResizeObserver | null>(null);
    const observedPlainRows = React.useRef<Set<Element>>(new Set());

    // Re-measure one row in place. The observer below calls this for a row that
    // resizes, and leaving the other rows' cached heights untouched keeps the
    // work proportional to the row that changed instead of the whole list.
    const measurePlainRowInPlace = React.useCallback((row: Element) => {
      const measured = measurePlainRow(row.parentElement ?? row);
      if (measured) plainItemHeights.current.set(measured[0], measured[1]);
    }, []);

    const getPlainRowObserver = React.useCallback(() => {
      if (plainRowObserver.current || typeof ResizeObserver === "undefined") {
        return plainRowObserver.current;
      }
      plainRowObserver.current = new ResizeObserver((entries) => {
        for (const entry of entries) {
          measurePlainRowInPlace(entry.target);
        }
      });
      return plainRowObserver.current;
    }, [measurePlainRowInPlace]);

    // Keep the cache current while the plain list is mounted: it seeds the
    // shell's geometry when the list later virtualizes. Only rows added since
    // the last run are measured and observed; a ResizeObserver then re-measures
    // a row that changes height *without* the row set changing — the reader
    // expands a group, or async content settles. Without that, a resize
    // followed by an append that crosses the threshold would seed a shell
    // shorter than the plain list it replaced, and the browser would clamp the
    // reader's scroll offset.
    React.useLayoutEffect(() => {
      // A list with no scroll container (shared/read-only views) can never
      // virtualize, so its cache is never read: measuring every row on every
      // append there is pure waste. And once the virtual shell has replaced the
      // plain list, its rows are detached: release the observer and its row set
      // so they do not retain that DOM for the rest of the conversation. The
      // cached heights survive, so returning to the plain phase reseeds it.
      if (scrollParent === undefined || shouldVirtualize) {
        plainRowObserver.current?.disconnect();
        plainRowObserver.current = null;
        observedPlainRows.current.clear();
        return undefined;
      }
      const list = plainListRef.current;
      if (!list) return undefined;

      // Observe each row's boxes, not the wrapper: the wrapper is
      // `display: contents` and has no box of its own.
      const observer = getPlainRowObserver();
      const currentRows = new Set<Element>();
      for (const child of Array.from(list.children)) {
        for (const row of Array.from(child.children)) {
          currentRows.add(row);
          if (observedPlainRows.current.has(row)) continue;
          const measured = measurePlainRow(child);
          if (measured) plainItemHeights.current.set(measured[0], measured[1]);
          observer?.observe(row);
          observedPlainRows.current.add(row);
        }
      }
      // Rows that left the list (collapsed history) must not keep the observer
      // alive on a detached node.
      observedPlainRows.current.forEach((row) => {
        if (currentRows.has(row)) return;
        observer?.unobserve(row);
        observedPlainRows.current.delete(row);
      });
      return undefined;
    }, [
      shouldVirtualize,
      scrollParent,
      rowKeys,
      getPlainRowObserver,
      measurePlainRowInPlace,
    ]);

    React.useEffect(
      () => () => {
        plainRowObserver.current?.disconnect();
        plainRowObserver.current = null;
        observedPlainRows.current.clear();
      },
      [],
    );

    // Seed the virtualizer once, the first time this list virtualizes, from the
    // heights the plain list just had. The value is captured in a ref rather
    // than recomputed: TanStack reads `initialMeasurementsCache` only while its
    // own cache is empty, so rebuilding it on every append after the switch
    // would be wasted work over a row set whose plain heights are no longer
    // updated.
    const initialMeasurementsCacheRef = React.useRef<VirtualItem[] | undefined>(
      undefined,
    );
    if (shouldVirtualize) {
      if (initialMeasurementsCacheRef.current === undefined) {
        initialMeasurementsCacheRef.current = buildInitialMeasurements(
          rowKeys,
          rowKeys.map((key) => plainItemHeights.current.get(key) ?? 0),
          ROW_GAP_PX,
        );
      }
    } else {
      // Reset while plain, so a later switch back to the shell re-seeds from
      // the heights current at that moment rather than a stale snapshot.
      initialMeasurementsCacheRef.current = undefined;
    }

    // Estimate for rows that mount after the switch (appended, or re-measured)
    // and so were never captured while plain. Using the flat
    // `ESTIMATED_ROW_HEIGHT` guess instead of the real rows' average height
    // biases the shell's own total: `chat-interface` restores the reader's
    // offset on "load older" from the shell's height delta, so a total built
    // from an over-tall estimate over-counts the prepended page and drags the
    // visible content up. Freeze the average of the rows measured while plain
    // (plus the row gap, which the measurement includes) once, since the
    // captured heights stop changing after the switch.
    const estimatedRowHeightRef = React.useRef<number | null>(null);
    if (shouldVirtualize) {
      if (estimatedRowHeightRef.current === null) {
        const heights = [...plainItemHeights.current.values()];
        estimatedRowHeightRef.current = heights.length
          ? Math.round(
              heights.reduce((total, height) => total + height, 0) /
                heights.length,
            ) + ROW_GAP_PX
          : ESTIMATED_ROW_HEIGHT;
      }
    } else {
      estimatedRowHeightRef.current = null;
    }

    const shellRef = React.useCallback(
      (node: HTMLDivElement | null) => {
        if (!node || !scrollParent) return undefined;
        const parent = node.parentElement;
        const measure = () => {
          const containerTop = scrollParent.getBoundingClientRect().top;
          const nodeTop = node.getBoundingClientRect().top;
          setScrollMargin(
            Math.round(nodeTop - containerTop + scrollParent.scrollTop),
          );
        };
        measure();
        // Preceding siblings can grow *after* mount (a /model card expands) or
        // appear late (the older-history spinner), so re-measure when their box
        // changes. Observing a node only tracks that node's own size, so a
        // MutationObserver watches the parent's child list too. A sibling that
        // appears *after* mount must also be observed: a late /model card that
        // later expands changes its height without adding a child, so a
        // one-shot observe at mount would never see the growth. Reconcile the
        // observed set on every child-list change. jsdom has neither observer;
        // the one-shot measurement above still runs there.
        const observers: { disconnect: () => void }[] = [];
        let resize: ResizeObserver | undefined;
        const observed = new Set<Element>();
        const reconcile = () => {
          measure();
          if (!resize) return;
          const siblings = new Set<Element>(
            Array.from(parent?.children ?? []).filter(
              (sibling) => sibling !== node,
            ),
          );
          siblings.forEach((sibling) => {
            if (!observed.has(sibling)) {
              resize?.observe(sibling);
              observed.add(sibling);
            }
          });
          observed.forEach((sibling) => {
            if (!siblings.has(sibling)) {
              resize?.unobserve(sibling);
              observed.delete(sibling);
            }
          });
        };
        if (typeof ResizeObserver !== "undefined") {
          resize = new ResizeObserver(measure);
          resize.observe(node);
          observers.push(resize);
          reconcile();
        }
        if (typeof MutationObserver !== "undefined" && parent) {
          const mutation = new MutationObserver(reconcile);
          mutation.observe(parent, { childList: true });
          observers.push(mutation);
        }
        // React 19 calls this cleanup when the ref detaches (unmount or a
        // changed ref identity), so the observers do not outlive the shell.
        return () => observers.forEach((observer) => observer.disconnect());
      },
      [scrollParent],
    );

    const rowVirtualizer = useVirtualizer({
      count: renderedItems.length,
      getScrollElement: () => scrollParent ?? null,
      getItemKey,
      estimateSize: () => estimatedRowHeightRef.current ?? ESTIMATED_ROW_HEIGHT,
      overscan: OVERSCAN,
      enabled: shouldVirtualize,
      // When `enabled` first flips on, TanStack seeds its internal offset from
      // `initialOffset` (default 0) and immediately scrolls the element there.
      // The virtualizer tracked nothing while it was disabled, so without this
      // the container snaps to the top on the threshold-crossing render — and
      // flashes there even when pinned to the bottom — before the chat's
      // stick-to-bottom pulls it back.
      initialOffset: () => scrollParent?.scrollTop ?? 0,
      // Offsets are relative to the scroll element, so tell the virtualizer
      // where this list begins inside it and shift each row back by the same
      // amount (see the transform below).
      scrollMargin,
      // Seed the first virtualized render with the plain list's real row
      // heights, so the shell's total height (and the reader's scroll offset)
      // is unchanged by the switch.
      initialMeasurementsCache: initialMeasurementsCacheRef.current,
      // Row heights are unknown until they mount (code blocks, images,
      // expanded output), so measure the live DOM and let the virtualizer
      // correct the offsets. The gap is added here because absolutely
      // positioned rows are outside the container's flex `gap`.
      measureElement: (element) =>
        Math.round(element.getBoundingClientRect().height) + ROW_GAP_PX,
    });

    // Keep the reader's place when a row is measured at a height other than its
    // estimate (see `anchorScrollOnItemSizeChange`). This is an instance
    // property, not a constructor option.
    rowVirtualizer.shouldAdjustScrollPositionOnItemSizeChange =
      anchorScrollOnItemSizeChange;

    // With a scroll container, rows can cross the virtualization threshold
    // and remount into the shell. Persist their expansion by row key in the
    // plain list too, so a control the user opened before the switch restores
    // once the shared shell mounts. Without a container (shared/read-only
    // views) component state is stable, so the Provider is omitted.
    const persistExpansion = scrollParent !== undefined;

    if (!shouldVirtualize) {
      return (
        <>
          {/*
            The wrapper is `display: contents`, so it adds no box: the rows keep
            the scroll column's own flex gap and their `last:` margins. It is
            only here to scope one correction — with a per-row wrapper each
            row's last element becomes a `:last-child`, so its `last:mb-4`
            would fire on every row instead of only the final one. The rule
            neutralizes that margin on every row but the last, and only on the
            elements that actually carry it.
          */}
          <div
            ref={plainListRef}
            className={cn("contents", PLAIN_ROW_MARGIN_RESET_CLASSNAME)}
          >
            {renderedItems.map((item) => {
              const itemKey = renderedItemKey(item);
              return (
                <div key={itemKey} data-row-key={itemKey} className="contents">
                  {persistExpansion ? (
                    <RowExpansionContext.Provider value={itemKey}>
                      {renderItem(item)}
                    </RowExpansionContext.Provider>
                  ) : (
                    renderItem(item)
                  )}
                </div>
              );
            })}
          </div>
          {secretRequests.unanchored.map((request) => (
            <SecretRequestCard key={request.toolCallId} request={request} />
          ))}
          <ConversationConfirmationButtons />
        </>
      );
    }

    return (
      <>
        <div
          ref={shellRef}
          data-testid="virtualized-message-list"
          style={{
            height: rowVirtualizer.getTotalSize(),
            width: "100%",
            position: "relative",
            // The rows are absolutely positioned, so this container has no
            // in-flow content and `min-height: auto` collapses to 0 — as a
            // flex child of the scrolling column it would then be shrunk back
            // to the viewport and the list could never scroll.
            flexShrink: 0,
          }}
        >
          {rowVirtualizer.getVirtualItems().map((virtualRow) => {
            const item = renderedItems[virtualRow.index];
            return (
              <div
                key={virtualRow.key}
                data-index={virtualRow.index}
                ref={rowVirtualizer.measureElement}
                data-testid="virtualized-message-row"
                // The scroll column is a flex column and the user bubble
                // right-aligns with `self-end`; an absolutely positioned row
                // would drop it out of that context, so the row reproduces the
                // column layout for its own children.
                //
                // `gap-2` restores the intra-row spacing the plain list gets
                // from the scroll column's own flex gap (its row wrapper is
                // `display: contents`, so a row's siblings are spaced by the
                // column). The margin correction mirrors the plain wrapper's
                // reset: each message is the last child of its own row, so its
                // `last:mb-4` would fire on every row instead of only the final
                // one. Scoped to the elements that carry `last:mb-4`, so it
                // leaves other margins (an `EventGroup`'s `my-1`) alone.
                className={cn(
                  "flex flex-col gap-2",
                  virtualRow.index < renderedItems.length - 1 &&
                    VIRTUAL_ROW_MARGIN_RESET_CLASSNAME,
                )}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  // `virtualRow.start` is measured from the scroll element
                  // and so includes the scroll margin; subtract it to place
                  // rows relative to this container.
                  transform: `translateY(${virtualRow.start - scrollMargin}px)`,
                }}
              >
                {/* Remounted rows read their expansion state back from the
                    store instead of starting collapsed. */}
                <RowExpansionContext.Provider value={renderedItemKey(item)}>
                  {renderItem(item)}
                </RowExpansionContext.Provider>
              </div>
            );
          })}
        </div>
        {secretRequests.unanchored.map((request) => (
          <SecretRequestCard key={request.toolCallId} request={request} />
        ))}
        <ConversationConfirmationButtons />
      </>
    );
  },
  (prevProps, nextProps) =>
    prevProps.messages.length === nextProps.messages.length &&
    prevProps.allEvents.length === nextProps.allEvents.length &&
    prevProps.scrollParent === nextProps.scrollParent &&
    getLastEventId(prevProps.messages) === getLastEventId(nextProps.messages) &&
    getLastEventId(prevProps.allEvents) ===
      getLastEventId(nextProps.allEvents) &&
    getLastEvent(prevProps.messages) === getLastEvent(nextProps.messages) &&
    getLastEvent(prevProps.allEvents) === getLastEvent(nextProps.allEvents),
);

Messages.displayName = "Messages";
