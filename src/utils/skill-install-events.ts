import { OpenHandsEvent } from "#/types/agent-server/core";
import { isExecuteBashObservationEvent } from "#/types/agent-server/type-guards";

export interface DetectedSkillInstall {
  /** Id of the bash observation event carrying the success marker. */
  eventId: string;
  skillName: string;
  /** Workspace root the skill was installed into (forward slashes). */
  workspacePath: string;
}

/**
 * Success marker printed by the bundled add-skill skill's fetch_skill.py:
 *   ✅ Successfully installed '<name>' to <workspace>/.agents/skills/<name>
 * Failures exit 1 and never print this line. Matched on output alone —
 * `command` can be "" and `exit_code` can be -1 on soft-timeout
 * continuation observations that still carry the full output.
 */
const INSTALL_SUCCESS_PATTERN =
  /^✅ Successfully installed '([^']+)' to (.+)$/m;

/**
 * Skills the agent installed during this conversation via the add-skill
 * flow. The SDK loads skills once at conversation start, so these are on
 * disk but inert until a new conversation starts in `workspacePath`.
 * Deduped per workspace+skill; a re-install (--force) keeps the latest
 * event id and moves the entry to the end.
 */
export function detectSkillInstalls(
  events: OpenHandsEvent[],
): DetectedSkillInstall[] {
  const byKey = new Map<string, DetectedSkillInstall>();

  for (const event of events) {
    if (!isExecuteBashObservationEvent(event)) continue;

    const text = event.observation.content
      .filter((c) => c.type === "text")
      .map((c) => c.text)
      .join("\n");
    const match = text.match(INSTALL_SUCCESS_PATTERN);
    if (!match) continue;

    const skillName = match[1];
    const dest = match[2].replace(/\\/g, "/").replace(/\/+$/, "");
    const suffix = `/.agents/skills/${skillName}`;
    if (!dest.endsWith(suffix)) continue;
    const workspacePath = dest.slice(0, -suffix.length);
    if (!workspacePath) continue;

    const key = `${workspacePath}::${skillName}`;
    byKey.delete(key);
    byKey.set(key, { eventId: event.id, skillName, workspacePath });
  }

  return [...byKey.values()];
}
