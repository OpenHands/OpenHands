import { describe, expect, it } from "vitest";
import { AUTOMATION_CATALOG } from "@openhands/extensions/automations";
import {
  OPENHANDS_CLOUD_INTEGRATIONS_URL,
  isResponderAutomation,
  resolveResponderDeploymentOption,
} from "#/utils/responder-deployment";

const automationById = (id: string) =>
  AUTOMATION_CATALOG.find((automation) => automation.id === id)!;

describe("isResponderAutomation", () => {
  it("matches pure GitHub/Slack automations and excludes multi-tool or other integrations", () => {
    // Arrange / Act / Assert — representative pure GitHub/Slack responders.
    expect(isResponderAutomation(automationById("github-pr-reviewer"))).toBe(
      true,
    );
    expect(isResponderAutomation(automationById("slack-channel-monitor"))).toBe(
      true,
    );
    expect(isResponderAutomation(automationById("slack-standup-digest"))).toBe(
      true,
    );
    // Multi-tool digest (slack + linear + notion) is not a pure responder.
    expect(
      isResponderAutomation(automationById("incident-retrospective-drafter")),
    ).toBe(false);
    // No GitHub/Slack integration at all.
    expect(
      isResponderAutomation(automationById("linear-triage-assistant")),
    ).toBe(false);
  });

  it("ignores an integration the automation can start without", () => {
    // Arrange — a GitHub responder that also publishes to Notion, but is
    // willing to run before Notion is connected.
    const automation = {
      ...automationById("github-pr-reviewer"),
      requires: {
        integrations: {
          github: { message: "Reads pull requests." },
          notion: { message: "Publishes a summary.", required: false as const },
        },
      },
    };

    // Act / Assert — where it runs is decided by what it needs.
    expect(isResponderAutomation(automation)).toBe(true);
  });
});

describe("resolveResponderDeploymentOption", () => {
  it("resolves each supported target to its action and rejects unsupported targets", () => {
    expect(resolveResponderDeploymentOption("local").action).toEqual({
      kind: "launch-local",
    });
    expect(resolveResponderDeploymentOption("openhands-cloud").action).toEqual({
      kind: "open-url",
      url: OPENHANDS_CLOUD_INTEGRATIONS_URL,
    });
    expect(OPENHANDS_CLOUD_INTEGRATIONS_URL).toBe(
      "https://app.all-hands.dev/settings/integrations",
    );
    expect(() => resolveResponderDeploymentOption("user-cloud")).toThrow();
  });
});
