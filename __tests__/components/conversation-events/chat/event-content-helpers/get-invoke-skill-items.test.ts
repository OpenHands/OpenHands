import { describe, expect, it } from "vitest";
import { getInvokeSkillItems } from "#/components/conversation-events/chat/event-content-helpers/get-invoke-skill-items";
import { ObservationEvent } from "#/types/agent-server/core";
import { InvokeSkillObservation } from "#/types/agent-server/core/base/observation";

const makeEvent = (
  observation: Partial<InvokeSkillObservation>,
): ObservationEvent<InvokeSkillObservation> =>
  ({
    id: "obs-skill",
    timestamp: "2026-06-08T00:00:00.000Z",
    source: "environment",
    tool_name: "invoke_skill",
    tool_call_id: "tool-skill",
    action_id: "action-skill",
    observation: {
      kind: "InvokeSkillObservation",
      skill_name: "worktree-switch",
      content: [],
      ...observation,
    },
  }) as ObservationEvent<InvokeSkillObservation>;

describe("getInvokeSkillItems", () => {
  it("returns a single item with the skill name and joined text content", () => {
    const items = getInvokeSkillItems(
      makeEvent({
        skill_name: "worktree-switch",
        content: [{ type: "text", text: "# Skill content" }],
      }),
    );

    expect(items).toEqual([
      { name: "worktree-switch", content: "# Skill content" },
    ]);
  });

  it("joins multiple text blocks with newlines and trims", () => {
    const items = getInvokeSkillItems(
      makeEvent({
        skill_name: "docker",
        content: [
          { type: "text", text: "  line one" },
          { type: "text", text: "line two  " },
        ],
      }),
    );

    expect(items).toEqual([{ name: "docker", content: "line one\nline two" }]);
  });

  it("ignores non-text content blocks", () => {
    const items = getInvokeSkillItems(
      makeEvent({
        skill_name: "docker",
        content: [
          { type: "image", image_urls: ["data:image/png;base64,abc"] },
          { type: "text", text: "only text survives" },
        ] as InvokeSkillObservation["content"],
      }),
    );

    expect(items).toEqual([{ name: "docker", content: "only text survives" }]);
  });

  it("returns an empty array when there is no skill name and no content", () => {
    const items = getInvokeSkillItems(
      makeEvent({ skill_name: "", content: [] }),
    );

    expect(items).toEqual([]);
  });
});
