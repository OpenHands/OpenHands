import React from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { ActionEvent, OpenHandsEvent } from "#/types/agent-server/core";
import {
  isActionEvent,
  isObservationEvent,
} from "#/types/agent-server/type-guards";
import { EventMessage } from "./event-message";
import { usePlanPreviewEvents } from "./hooks/use-plan-preview-events";
import { groupEvents, renderedItemKey, RenderedItem } from "./group-events";
import { EventGroup } from "./event-message-components/event-group";
import { ThoughtEventMessage } from "./event-message-components/thought-event-message";
import { useModelStore } from "#/stores/model-store";
import { ModelMessages } from "#/components/features/chat/model-messages";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { ConversationConfirmationButtons } from "#/components/shared/buttons/conversation-confirmation-buttons";
import { RowExpansionContext } from "#/components/features/chat/row-expansion-context";
import { useMessageExpansionStore } from "#/stores/message-expansion-store";

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
 */
const VIRTUALIZATION_THRESHOLD = 150;
/**
 * Dev-only escape hatch used by the capture harness in `.tmp/`: `?virtualize=0`
 * raises the threshold so the same conversation can be filmed with and without
 * virtualization. Never set in production builds, so behavior is unchanged.
 */
const virtualizationThreshold = () => {
  if (!import.meta.env.DEV || typeof window === "undefined") {
    return VIRTUALIZATION_THRESHOLD;
  }
  const flag = new URLSearchParams(window.location.search).get("virtualize");
  return flag === "0" ? Number.POSITIVE_INFINITY : VIRTUALIZATION_THRESHOLD;
};
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
    const actionById = React.useMemo(() => {
      const actions = new Map<string, ActionEvent>();
      for (const event of allEvents) {
        if (isActionEvent(event)) actions.set(event.id, event);
      }
      return actions;
    }, [allEvents]);

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

    // Fold consecutive action/observation events into collapsible groups so a
    // long sequence of tool calls doesn't dominate the chat scroll. Items that
    // can't be grouped (or that fall in a short run) are still rendered one by
    // one, identically to before. Agent thoughts attached to an action are
    // hoisted out as their own rendered item so they always show up in the
    // message pane and a thought between actions starts a fresh group.
    const renderedItems = React.useMemo(
      () => groupEvents(messages, undefined, allEvents),
      [messages, allEvents],
    );

    const renderEventMessage = (
      event: OpenHandsEvent,
      index: number,
      suppressThought: boolean,
    ) => (
      <EventMessage
        key={event.id}
        event={event}
        correspondingAction={
          isObservationEvent(event) && event.action_id
            ? (actionById.get(event.action_id) ?? null)
            : null
        }
        isLastMessage={messages.length - 1 === index}
        isInLast10Actions={messages.length - 1 - index < 10}
        showPlanPreview={event.id ? planPreviewEventIds.has(event.id) : false}
        suppressThought={suppressThought}
      />
    );

    const renderItem = (item: RenderedItem, itemIndex: number) => {
      if (item.kind === "single") {
        return (
          <>
            {/* Thoughts for singles are also hoisted as their own
                "thought" item, so suppress the inline render to avoid
                duplication. */}
            {renderEventMessage(item.event, item.index, true)}
            {maybeRenderModelMessages(item.event.id)}
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
      const isFinalized = itemIndex < renderedItems.length - 1;
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
            </React.Fragment>
          ))}
        </>
      );
    };

    // `null` (not `undefined`) means the caller has a scroll container but the
    // element is not mounted yet: virtualize the shell rather than rendering
    // every row for a frame.
    const shouldVirtualize = Boolean(
      scrollParent !== undefined &&
      renderedItems.length >= virtualizationThreshold(),
    );

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

    // The virtualizer's scroll element is the chat column, which also holds
    // content *before* this list (top-anchored /model cards, the older-history
    // spinner). TanStack compares the container's `scrollTop` against item
    // starts, so without the list's own offset it believes the list starts at
    // 0 and mounts the wrong rows — leaving blank space near the top once the
    // preceding content is taller than the overscan. Measure the shell's
    // offset within the scroll element and pass it as `scrollMargin`.
    const [scrollMargin, setScrollMargin] = React.useState(0);
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
      getItemKey: (index) => renderedItemKey(renderedItems[index]),
      estimateSize: () => ESTIMATED_ROW_HEIGHT,
      overscan: OVERSCAN,
      enabled: shouldVirtualize,
      // Offsets are relative to the scroll element, so tell the virtualizer
      // where this list begins inside it and shift each row back by the same
      // amount (see the transform below).
      scrollMargin,
      // Row heights are unknown until they mount (code blocks, images,
      // expanded output), so measure the live DOM and let the virtualizer
      // correct the offsets. The gap is added here because absolutely
      // positioned rows are outside the container's flex `gap`.
      measureElement: (element) =>
        Math.round(element.getBoundingClientRect().height) + ROW_GAP_PX,
    });

    if (!shouldVirtualize) {
      return (
        <>
          {renderedItems.map((item, itemIndex) => (
            <React.Fragment key={renderedItemKey(item)}>
              {renderItem(item, itemIndex)}
            </React.Fragment>
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
                  {renderItem(item, virtualRow.index)}
                </RowExpansionContext.Provider>
              </div>
            );
          })}
        </div>
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
