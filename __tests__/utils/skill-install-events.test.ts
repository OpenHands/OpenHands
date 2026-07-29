import { describe, expect, it } from "vitest";
import { detectSkillInstalls } from "#/utils/skill-install-events";
import type {
  CmdOutputMetadata,
  ExecuteBashObservation,
  ObservationEvent,
} from "#/types/agent-server/core";
import { createUserMessageEvent } from "test-utils";

const successLine = (skill: string, dest: string) =>
  `Fetching skill '${skill}'...\n✅ Successfully installed '${skill}' to ${dest}`;

const makeBashObservationEvent = (
  id: string,
  text: string,
  overrides: Partial<ExecuteBashObservation> = {},
): ObservationEvent<ExecuteBashObservation> => ({
  id,
  timestamp: new Date().toISOString(),
  source: "environment",
  tool_name: "execute_bash",
  tool_call_id: `call-${id}`,
  action_id: `action-${id}`,
  observation: {
    kind: "ExecuteBashObservation",
    content: [{ type: "text", text }],
    command:
      'python3 /skills/add-skill/scripts/fetch_skill.py "https://github.com/o/r/tree/main/skills/s" "/ws"',
    exit_code: 0,
    error: false,
    timeout: false,
    metadata: {} as CmdOutputMetadata,
    ...overrides,
  },
});

describe("detectSkillInstalls", () => {
  it("detects a successful install and extracts the skill name, workspace, and event id", () => {
    const events = [
      createUserMessageEvent("evt-msg"),
      makeBashObservationEvent(
        "evt-install",
        successLine("codereview", "/tmp/demo-ws/.agents/skills/codereview"),
      ),
    ];

    const installs = detectSkillInstalls(events);

    expect(installs).toEqual([
      {
        eventId: "evt-install",
        skillName: "codereview",
        workspacePath: "/tmp/demo-ws",
      },
    ]);
  });

  it("ignores bash output without the success marker", () => {
    const events = [
      makeBashObservationEvent(
        "evt-fail",
        "❌ No SKILL.md found in 'skills/broken' - not a valid skill",
        { exit_code: 1, error: true },
      ),
    ];

    expect(detectSkillInstalls(events)).toEqual([]);
  });

  it("ignores a success line whose destination is not a .agents/skills path", () => {
    const events = [
      makeBashObservationEvent(
        "evt-odd",
        successLine("codereview", "/tmp/elsewhere/codereview"),
      ),
    ];

    expect(detectSkillInstalls(events)).toEqual([]);
  });

  it("detects an install on a soft-timeout continuation observation", () => {
    const events = [
      makeBashObservationEvent(
        "evt-timeout",
        successLine("codereview", "/tmp/demo-ws/.agents/skills/codereview"),
        { command: "", exit_code: -1 },
      ),
    ];

    expect(detectSkillInstalls(events)).toHaveLength(1);
  });

  it("keeps only the latest event for a re-installed skill", () => {
    const dest = "/tmp/demo-ws/.agents/skills/codereview";
    const events = [
      makeBashObservationEvent("evt-first", successLine("codereview", dest)),
      makeBashObservationEvent("evt-second", successLine("codereview", dest)),
    ];

    const installs = detectSkillInstalls(events);

    expect(installs).toEqual([
      expect.objectContaining({ eventId: "evt-second" }),
    ]);
  });

  it("normalizes Windows-style destination paths", () => {
    const events = [
      makeBashObservationEvent(
        "evt-win",
        successLine("codereview", "C:\\ws\\.agents\\skills\\codereview"),
      ),
    ];

    expect(detectSkillInstalls(events)).toEqual([
      expect.objectContaining({ workspacePath: "C:/ws" }),
    ]);
  });
});
