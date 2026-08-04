import type {
  AgentNotification,
  AgentNotificationKind,
} from "./agent-notifications.constants";

/**
 * Fenced code-block language tag the detection skill uses to hand a
 * candidate skill/automation back to the UI, e.g.:
 *
 * ```agent-notification
 * {"id": "...", "kind": "skill", "name": "...", "prompt": "..."}
 * ```
 *
 * See `.agents/skills/reusable-extension-detector.md` for the full contract.
 */
export const AGENT_NOTIFICATION_FENCE_LANGUAGE = "agent-notification";

const FENCE_REGEX = /```agent-notification\s*\n([\s\S]*?)\n```/g;

const VALID_KINDS: readonly AgentNotificationKind[] = [
  "skill",
  "workflow",
  "routine",
  "responder",
];

function isValidKind(value: unknown): value is AgentNotificationKind {
  return (
    typeof value === "string" &&
    (VALID_KINDS as readonly string[]).includes(value)
  );
}

function parseBlock(raw: string): AgentNotification | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }

  if (typeof parsed !== "object" || parsed === null) {
    return null;
  }

  const { id, kind, name, prompt } = parsed as Record<string, unknown>;

  if (
    typeof id !== "string" ||
    !id.trim() ||
    typeof name !== "string" ||
    !name.trim() ||
    typeof prompt !== "string" ||
    !prompt.trim() ||
    !isValidKind(kind)
  ) {
    return null;
  }

  return {
    id: id.trim(),
    kind,
    name: name.trim(),
    prompt: prompt.trim(),
    createdAt: new Date().toISOString(),
  };
}

/**
 * Strips `agent-notification` fenced blocks out of assistant message text
 * (so they never render as a raw code block) and returns the parsed,
 * validated notifications found within them. Malformed blocks are dropped
 * silently rather than surfaced to the user.
 */
export function extractAgentNotifications(text: string): {
  message: string;
  notifications: AgentNotification[];
} {
  if (!text.includes(AGENT_NOTIFICATION_FENCE_LANGUAGE)) {
    return { message: text, notifications: [] };
  }

  const notifications: AgentNotification[] = [];
  const message = text
    .replace(FENCE_REGEX, (_match, body: string) => {
      const notification = parseBlock(body.trim());
      if (notification) {
        notifications.push(notification);
      }
      return "";
    })
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  return { message, notifications };
}
