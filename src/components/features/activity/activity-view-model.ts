import { I18nKey } from "#/i18n/declaration";
import i18n from "#/i18n";
import type {
  AppConversation,
  MetricsSnapshot,
} from "#/api/conversation-service/agent-server-conversation-service.types";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import type { MessageEvent, OpenHandsEvent } from "#/types/agent-server/core";
import {
  isActionEvent,
  isMessageEvent,
  isObservationEvent,
} from "#/types/agent-server/type-guards";
import {
  getActionEventTitleDescriptor,
  type EventTitleDescriptor,
} from "#/components/conversation-events/chat/event-content-helpers/get-action-event-title";

// @spec LAV-001 — Only actively executing agents are listed

export type ActivityStatusKind =
  | "running"
  | "waiting"
  | "paused"
  | "error"
  | "finished"
  | "idle"
  | "unknown";

export interface ActivityStatusDescriptor {
  kind: ActivityStatusKind;
  /** Localized label for the status chip. */
  labelKey: I18nKey;
  /** True for states a user must act on (confirmation, error, stuck). */
  needsAttention: boolean;
}

/**
 * Map an `ExecutionStatus` onto the activity view's presentation model.
 * `needsAttention` is the visual contract that separates "running" from
 * "blocked": waiting-for-confirmation and error/stuck demand user action,
 * while running/paused/finished/idle do not.
 */
export function getActivityStatusDescriptor(
  status: ExecutionStatus | null | undefined,
): ActivityStatusDescriptor {
  switch (status) {
    case ExecutionStatus.RUNNING:
      return {
        kind: "running",
        labelKey: I18nKey.ACTIVITY$STATUS_RUNNING,
        needsAttention: false,
      };
    case ExecutionStatus.WAITING_FOR_CONFIRMATION:
      return {
        kind: "waiting",
        labelKey: I18nKey.ACTIVITY$STATUS_WAITING,
        needsAttention: true,
      };
    case ExecutionStatus.PAUSED:
      return {
        kind: "paused",
        labelKey: I18nKey.ACTIVITY$STATUS_PAUSED,
        needsAttention: false,
      };
    case ExecutionStatus.ERROR:
    case ExecutionStatus.STUCK:
      return {
        kind: "error",
        labelKey: I18nKey.ACTIVITY$STATUS_ERROR,
        needsAttention: true,
      };
    case ExecutionStatus.FINISHED:
      return {
        kind: "finished",
        labelKey: I18nKey.ACTIVITY$STATUS_FINISHED,
        needsAttention: false,
      };
    case ExecutionStatus.IDLE:
      return {
        kind: "idle",
        labelKey: I18nKey.ACTIVITY$STATUS_IDLE,
        needsAttention: false,
      };
    default:
      return {
        kind: "unknown",
        labelKey: I18nKey.ACTIVITY$STATUS_UNKNOWN,
        needsAttention: false,
      };
  }
}

/**
 * A conversation belongs in the live list only while it is actively
 * executing. Idle, paused, finished, error and unknown are excluded here and
 * shown through the conversation list instead.
 */
export function isActiveActivityStatus(
  status: ExecutionStatus | null | undefined,
): boolean {
  return (
    status === ExecutionStatus.RUNNING ||
    status === ExecutionStatus.WAITING_FOR_CONFIRMATION
  );
}

export function selectActiveConversations(
  conversations: AppConversation[],
): AppConversation[] {
  return conversations.filter((conversation) =>
    isActiveActivityStatus(conversation.execution_status),
  );
}

export type SubagentStatus = "running" | "completed" | "error";

export interface SubagentActivity {
  /** The `task` tool call id, exposed as the delegation's public id. */
  id: string;
  /** The specialized subagent the parent delegated to. */
  name: string;
  status: SubagentStatus;
}

/**
 * Derive the in-process `task`-tool delegations for one conversation from its
 * event stream. A `TaskAction` opens a delegation; the matching
 * `TaskObservation` closes it as completed or errored. Delegations with no
 * observation yet stay "running".
 *
 * Pairing is by event identity, not by tool-call id: a `TaskObservation`
 * references its action through `action_id` (an event id), while the action's
 * `tool_call_id` is a separate value. Delegations are therefore keyed by the
 * action event's `id`, and `tool_call_id` is only the public id handed to the
 * row.
 */
// @spec LAV-003 — Subagent fan-out is derived from the event stream
export function deriveSubagents(events: OpenHandsEvent[]): SubagentActivity[] {
  const order: string[] = [];
  const byActionEventId = new Map<string, SubagentActivity>();

  for (const event of events) {
    if (isActionEvent(event) && event.action.kind === "TaskAction") {
      if (!byActionEventId.has(event.id)) {
        order.push(event.id);
        byActionEventId.set(event.id, {
          id: event.tool_call_id,
          name: event.action.subagent_type,
          status: "running",
        });
      }
    } else if (
      isObservationEvent(event) &&
      event.observation.kind === "TaskObservation"
    ) {
      const entry = byActionEventId.get(event.action_id);
      if (entry) {
        entry.status = event.observation.is_error ? "error" : "completed";
      }
    }
  }

  return order
    .map((actionEventId) => byActionEventId.get(actionEventId))
    .filter((entry): entry is SubagentActivity => entry !== undefined);
}

/** Action event ids that a `TaskObservation` in `events` has resolved. */
function resolvedTaskActionEventIds(events: OpenHandsEvent[]): Set<string> {
  const resolved = new Set<string>();
  for (const event of events) {
    if (
      isObservationEvent(event) &&
      event.observation.kind === "TaskObservation"
    ) {
      resolved.add(event.action_id);
    }
  }
  return resolved;
}

/** `task` actions in `events` that no `TaskObservation` has resolved yet. */
function unresolvedTaskActions(events: OpenHandsEvent[]): OpenHandsEvent[] {
  const resolved = resolvedTaskActionEventIds(events);

  return events.filter(
    (event) =>
      isActionEvent(event) &&
      event.action.kind === "TaskAction" &&
      !resolved.has(event.id),
  );
}

/**
 * The `TaskObservation` events that resolved a delegation, retained
 * independently of the bounded event history. Without them, a completed
 * delegation whose observation scrolls out of the window would be re-derived
 * as still running once its action is carried back in.
 */
function mergeResolvedObservations(
  previous: OpenHandsEvent[] | undefined,
  next: OpenHandsEvent[],
): OpenHandsEvent[] {
  const byActionId = new Map<string, OpenHandsEvent>();
  for (const event of previous ?? []) {
    if (
      isObservationEvent(event) &&
      event.observation.kind === "TaskObservation"
    ) {
      byActionId.set(event.action_id, event);
    }
  }
  for (const event of next) {
    if (
      isObservationEvent(event) &&
      event.observation.kind === "TaskObservation"
    ) {
      byActionId.set(event.action_id, event);
    }
  }
  return [...byActionId.values()];
}

/** Action event ids of the task actions in `events`. */
function taskActionEventIds(events: OpenHandsEvent[]): Set<string> {
  const ids = new Set<string>();
  for (const event of events) {
    if (isActionEvent(event) && event.action.kind === "TaskAction") {
      ids.add(event.id);
    }
  }
  return ids;
}

/** Action event ids that a `TaskObservation` in `events` has resolved. */
function observedActionIds(events: OpenHandsEvent[]): Set<string> {
  const ids = new Set<string>();
  for (const event of events) {
    if (
      isObservationEvent(event) &&
      event.observation.kind === "TaskObservation"
    ) {
      ids.add(event.action_id);
    }
  }
  return ids;
}

/**
 * Re-add the retained observations for task actions whose own observation fell
 * out of the bounded history, so `deriveSubagents` sees each retained action's
 * true terminal state instead of reopening it as running.
 */
function withResolvedObservations(
  events: OpenHandsEvent[],
  resolvedObservations: OpenHandsEvent[],
): OpenHandsEvent[] {
  if (resolvedObservations.length === 0) return events;

  const actionIds = taskActionEventIds(events);
  const observed = observedActionIds(events);
  const extras: OpenHandsEvent[] = [];
  for (const observation of resolvedObservations) {
    if (!isObservationEvent(observation)) continue;
    if (
      !actionIds.has(observation.action_id) ||
      observed.has(observation.action_id)
    ) {
      continue;
    }
    observed.add(observation.action_id);
    extras.push(observation);
  }

  return extras.length > 0 ? [...events, ...extras] : events;
}

/**
 * Drop retained resolutions whose task action is no longer present. A resolved
 * action is never carried, so once it leaves the bounded history its resolution
 * can never be observed again; pruning keeps the retention bounded rather than
 * growing for the lifetime of the conversation.
 */
function pruneResolvedObservations(
  resolvedObservations: OpenHandsEvent[],
  actionIds: Set<string>,
): OpenHandsEvent[] {
  return resolvedObservations.filter(
    (event) => isObservationEvent(event) && actionIds.has(event.action_id),
  );
}

/**
 * A bounded window of one conversation's events plus the polling watermark.
 *
 * `watermark` is the timestamp of the newest event seen so far. The next poll
 * asks the backend for events at or after it, which is what makes the poll
 * gapless: an observation can no longer slip between two polls and be missed,
 * so an unresolved delegation is provably still running.
 */
export interface ActivityTailBuffer {
  events: OpenHandsEvent[];
  /**
   * The `TaskObservation` that resolved each delegation, retained outside the
   * bounded `events` window so a completed action carried back in still reads
   * as completed rather than running.
   */
  resolvedObservations?: OpenHandsEvent[];
  watermark?: string;
  /**
   * Page cursor to resume an unfinished timestamp-filtered range. Set when a
   * poll hit the page bound before exhausting the range, so the next poll
   * continues from where it stopped instead of re-reading the newest pages.
   */
  resumePageId?: string;
  /**
   * Set once the backend has rejected a timestamp-filtered request, so later
   * polls skip the attempt and read a plain tail instead.
   */
  supportsTimestampFilter?: boolean;
}

/**
 * How many events a conversation keeps across polls. Completed delegations
 * linger in this window so the row does not flicker back to zero subagents the
 * moment a task finishes.
 */
export const ACTIVITY_TAIL_HISTORY_LIMIT = 60;

function eventTimestampMs(event: OpenHandsEvent): number | null {
  const timestamp = event.timestamp;
  if (typeof timestamp !== "string") return null;

  const parsed = Date.parse(timestamp);
  return Number.isNaN(parsed) ? null : parsed;
}

/** The later of two ISO timestamps, tolerating missing or unparseable values. */
function maxTimestamp(a?: string, b?: string): string | undefined {
  if (a === undefined) return b;
  if (b === undefined) return a;

  const aMs = Date.parse(a);
  const bMs = Date.parse(b);
  if (Number.isNaN(aMs)) return b;
  if (Number.isNaN(bMs)) return a;

  return aMs >= bMs ? a : b;
}

/** Timestamp of the newest event in `events`, or undefined when none parse. */
export function latestEventTimestamp(
  events: OpenHandsEvent[],
): string | undefined {
  let newest: string | undefined;
  for (const event of events) {
    if (typeof event.timestamp === "string") {
      newest = maxTimestamp(newest, event.timestamp);
    }
  }
  return newest;
}

function compareByTimestamp(a: OpenHandsEvent, b: OpenHandsEvent): number {
  const aMs = eventTimestampMs(a);
  const bMs = eventTimestampMs(b);

  if (aMs === null && bMs === null) return 0;
  if (aMs === null) return 1;
  if (bMs === null) return -1;
  return aMs - bMs;
}

export interface MergeActivityTailOptions {
  /**
   * True only when the caller fetched every event since the previous
   * watermark. An unresolved delegation is then known to be still running and
   * may be carried forward. When the window may contain gaps (no watermark
   * yet, or a backend without timestamp filters) carried delegations are
   * dropped instead of being reported as running indefinitely.
   */
  canIncrementallyFetch?: boolean;
  /**
   * True when `next` holds every event in the requested range, so the newest
   * timestamp in it is a valid watermark for the next poll. A partial page
   * (the range was longer than one page and pagination stopped early) must not
   * advance the watermark: doing so would permanently exclude the events the
   * next poll never asked for.
   */
  rangeComplete?: boolean;
  /**
   * Page cursor for the unfinished remainder of the range when it was not
   * complete. Stored so the next poll resumes from there instead of re-reading
   * the newest pages of a large backlog.
   */
  resumePageId?: string;
  /** Records that the backend rejected a timestamp-filtered request. */
  supportsTimestampFilter?: boolean;
}

/**
 * Merge a freshly fetched window into the previous buffer.
 *
 * The fetched window is deduplicated into a bounded, timestamp-ordered history
 * so `pickLatestActivity` and `deriveSubagents` still see the delegations that
 * finished recently. A `task` action with no observation is carried forward
 * only when the caller proved the window is gapless; keeping it also lets a
 * later `TaskObservation` pair with it even though the action itself has
 * scrolled out.
 */
// @spec LAV-004 — Data is bounded and read-only
export function mergeActivityTail(
  previous: ActivityTailBuffer | undefined,
  next: OpenHandsEvent[],
  options: MergeActivityTailOptions = {},
): ActivityTailBuffer {
  const canIncrementallyFetch = options.canIncrementallyFetch === true;
  const rangeComplete = options.rangeComplete !== false;
  const base = canIncrementallyFetch ? (previous?.events ?? []) : [];

  // Streaming/state events carry no `id` and are irrelevant to the current
  // step and the delegation fan-out, so only identified events are buffered.
  const byId = new Map<string, OpenHandsEvent>();
  for (const event of base) {
    if (typeof event.id === "string") byId.set(event.id, event);
  }
  for (const event of next) {
    if (typeof event.id === "string") byId.set(event.id, event);
  }

  const ordered = [...byId.values()].sort(compareByTimestamp);
  const history =
    ordered.length > ACTIVITY_TAIL_HISTORY_LIMIT
      ? ordered.slice(-ACTIVITY_TAIL_HISTORY_LIMIT)
      : ordered;
  const historyIds = new Set(history.map((event) => event.id));
  // A carried action whose observation was displaced from the bounded history
  // would stay "running" forever, so only carry an action while the whole
  // range since the watermark was actually read (the observation cannot have
  // slipped by unseen).
  const carried = rangeComplete
    ? unresolvedTaskActions(base).filter((event) => !historyIds.has(event.id))
    : [];

  // Only advance the watermark over a complete range. On a partial page the
  // newest event is not a safe lower bound for the next poll: events between
  // the watermark and it were never requested, and advancing past them would
  // drop them permanently.
  const watermark = rangeComplete
    ? maxTimestamp(latestEventTimestamp(next), previous?.watermark)
    : previous?.watermark;
  const supportsTimestampFilter =
    options.supportsTimestampFilter === false ||
    previous?.supportsTimestampFilter === false;

  // Keep the resolutions separate from the bounded history so a carried action
  // still reads as completed (or errored) after its observation scrolls out.
  // Prune resolutions whose action is gone so the retention stays bounded.
  const baseEvents = carried.length > 0 ? [...carried, ...history] : history;
  const resolvedObservations = pruneResolvedObservations(
    mergeResolvedObservations(
      canIncrementallyFetch ? previous?.resolvedObservations : undefined,
      next,
    ),
    taskActionEventIds(baseEvents),
  );
  const events = withResolvedObservations(baseEvents, resolvedObservations);

  return {
    events,
    ...(resolvedObservations.length > 0 ? { resolvedObservations } : {}),
    ...(watermark !== undefined ? { watermark } : {}),
    ...(rangeComplete
      ? {}
      : options.resumePageId !== undefined
        ? { resumePageId: options.resumePageId }
        : {}),
    ...(supportsTimestampFilter ? { supportsTimestampFilter: false } : {}),
  };
}

function messageText(event: MessageEvent): string | null {
  const content = event.llm_message.content;
  if (!Array.isArray(content)) return null;
  const text = content
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("\n")
    .trim();
  return text || null;
}

/**
 * The newest thing a conversation is doing: its most recent action/tool or
 * assistant message, whichever is later in the stream, rendered through the
 * shared action-title descriptor. Returns null when neither exists so the row
 * can show a defined "no activity" state instead of a blank.
 *
 * The tail is chronological, so a single reverse scan is what makes "latest"
 * true: an earlier action must not win over a later assistant message (and
 * vice versa). A separate scan for actions would always prefer an old action.
 */
export function pickLatestActivity(
  events: OpenHandsEvent[],
): EventTitleDescriptor | null {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const event = events[i];

    if (isActionEvent(event) && event.tool_name) {
      return getActionEventTitleDescriptor(event);
    }

    if (isMessageEvent(event) && event.llm_message.role === "assistant") {
      const text = messageText(event);
      if (text) {
        return { kind: "text", text };
      }
    }
  }

  return null;
}

export interface ActivityMetrics {
  /** Accumulated cost, or null when the backend reported no metrics. */
  cost: number | null;
  /** Prompt + completion tokens, or null when usage is unavailable. */
  totalTokens: number | null;
}

export function getActivityMetrics(
  metrics: MetricsSnapshot | null | undefined,
): ActivityMetrics {
  if (!metrics) {
    return { cost: null, totalTokens: null };
  }

  const usage = metrics.accumulated_token_usage;
  return {
    cost: metrics.accumulated_cost ?? null,
    totalTokens: usage ? usage.prompt_tokens + usage.completion_tokens : null,
  };
}

/**
 * Resolve an action-title descriptor to a plain, localized string for the row.
 * The shared helper's translation templates embed styling-only component tags
 * (`<cmd>`, `<path>`, …) that `getEventContent` renders through `<Trans>`; the
 * activity row needs text inside a link, so the tags are stripped and only
 * their inner content is kept.
 */
export function resolveDescriptorText(
  descriptor: EventTitleDescriptor,
): string {
  if (descriptor.kind === "text") {
    return descriptor.text;
  }

  if (!i18n.exists(descriptor.key)) {
    return descriptor.key;
  }

  return i18n.t(descriptor.key, descriptor.values).replace(/<[^>]+>/g, "");
}
