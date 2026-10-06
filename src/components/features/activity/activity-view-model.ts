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
  /** The `task` tool call id, used to pair the action with its observation. */
  id: string;
  /** The specialized subagent the parent delegated to. */
  name: string;
  status: SubagentStatus;
}

/**
 * Derive the in-process `task`-tool delegations for one conversation from its
 * event stream. A `TaskAction` opens a delegation keyed by its tool-call id;
 * the matching `TaskObservation` (paired by `action_id`) closes it as
 * completed or errored. Delegations with no observation yet stay "running".
 */
// @spec LAV-003 — Subagent fan-out is derived from the event stream
export function deriveSubagents(events: OpenHandsEvent[]): SubagentActivity[] {
  const order: string[] = [];
  const byId = new Map<string, SubagentActivity>();

  for (const event of events) {
    if (isActionEvent(event) && event.action.kind === "TaskAction") {
      const id = event.tool_call_id;
      if (!byId.has(id)) {
        order.push(id);
        byId.set(id, {
          id,
          name: event.action.subagent_type,
          status: "running",
        });
      }
    } else if (
      isObservationEvent(event) &&
      event.observation.kind === "TaskObservation"
    ) {
      const entry = byId.get(event.action_id);
      if (entry) {
        entry.status = event.observation.is_error ? "error" : "completed";
      }
    }
  }

  return order
    .map((id) => byId.get(id))
    .filter((entry): entry is SubagentActivity => entry !== undefined);
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
 * The newest thing a conversation is doing: its most recent action/tool
 * rendered through the shared action-title descriptor, falling back to the
 * last assistant message text. Returns null when neither exists so the row
 * can show a defined "no activity" state instead of a blank.
 */
export function pickLatestActivity(
  events: OpenHandsEvent[],
): EventTitleDescriptor | null {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const event = events[i];
    if (isActionEvent(event) && event.tool_name) {
      return getActionEventTitleDescriptor(event);
    }
  }

  for (let i = events.length - 1; i >= 0; i -= 1) {
    const event = events[i];
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
