import { describe, expect, it } from "vitest";
import {
  buildAutomationEventTrigger,
  parseAutomationEventKeys,
} from "#/api/automation-setup-events";

describe("automation event editing", () => {
  it("preserves custom and wildcard keys in a comma-separated agent edit", () => {
    const keys = parseAutomationEventKeys(
      " pr:opened, pr:comment:*, custom[12], pr:opened ",
    );
    expect(keys).toEqual(["pr:opened", "pr:comment:*", "custom[12]"]);
    expect(
      buildAutomationEventTrigger("bitbucket_data_center", keys.join(", "), "")
        .on,
    ).toEqual(keys);
  });

  it("retains wildcard scope and the trigger extras supplied by the edit draft", () => {
    const trigger = buildAutomationEventTrigger("github", "*", "", {
      type: "event",
      source: "github",
      on: "issues.opened",
      destination: "continue_conversation",
    });
    expect(trigger).toEqual({
      type: "event",
      source: "github",
      on: "*",
      destination: "continue_conversation",
    });
  });

  it("preserves advanced filter text when changing event keys", () => {
    const filter =
      "  contains(labels[].name, 'ready') || author.login == 'bot'  ";
    expect(
      buildAutomationEventTrigger(
        "github",
        "issues.opened, issues.closed",
        filter,
      ).filter,
    ).toBe(filter);
  });

  it("leaves an empty selection empty for validation instead of selecting an event", () => {
    expect(buildAutomationEventTrigger("github", "", "").on).toEqual([]);
  });
});
