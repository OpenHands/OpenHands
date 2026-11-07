import { describe, expect, it } from "vitest";

import type { DeploymentCapabilities } from "#/manifests/types";
import {
  eventKeyOptionsForSource,
  fallbackEventKeyLabel,
} from "./automation-event-sources";

const GITHUB_CAPABILITIES: DeploymentCapabilities = {
  ready: true,
  triggerKinds: ["cron", "event"],
  eventSources: ["github"],
  eventTypes: ["issues.*", "pull_request.*", "push"],
  eventLabelsBySource: {
    github: {
      "pull_request.*": "Pull request activity",
      push: "Push",
    },
  },
  triggers: {},
  features: [],
};

describe("fallbackEventKeyLabel", () => {
  it("names a family pattern and its any-action wildcard", () => {
    expect(fallbackEventKeyLabel("pull_request.*")).toBe(
      "Pull request (any action)",
    );
  });

  it("expands the pull request abbreviation and separators", () => {
    expect(fallbackEventKeyLabel("pr:opened")).toBe("Pull request opened");
    expect(fallbackEventKeyLabel("git.pullrequest.created")).toBe(
      "Git pull request created",
    );
  });

  it("capitalizes simple keys without inventing words", () => {
    expect(fallbackEventKeyLabel("jira:issue_created")).toBe(
      "Jira issue created",
    );
    expect(fallbackEventKeyLabel("push")).toBe("Push");
  });
});

describe("eventKeyOptionsForSource", () => {
  it("uses the deployment's label for each pattern", () => {
    expect(eventKeyOptionsForSource(GITHUB_CAPABILITIES, "github")).toEqual([
      { key: "issues.*", label: "Issues (any action)" },
      { key: "pull_request.*", label: "Pull request activity" },
      { key: "push", label: "Push" },
    ]);
  });

  it("falls back to a prettified key where no label is published", () => {
    const capabilities = { ...GITHUB_CAPABILITIES };
    delete capabilities.eventLabelsBySource;

    expect(eventKeyOptionsForSource(capabilities, "github")).toEqual([
      { key: "issues.*", label: "Issues (any action)" },
      { key: "pull_request.*", label: "Pull request (any action)" },
      { key: "push", label: "Push" },
    ]);
  });

  it("offers nothing for a source without a catalog", () => {
    expect(
      eventKeyOptionsForSource(
        { ...GITHUB_CAPABILITIES, eventTypesBySource: undefined },
        "github",
      ).filter((option) => option.key === "nonexistent"),
    ).toEqual([]);
    expect(eventKeyOptionsForSource(null, "github")).toEqual([]);
  });
});
