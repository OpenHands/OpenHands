import { describe, expect, it } from "vitest";
import { AUTOMATION_CATALOG } from "@openhands/extensions/automations";
import { SETUP_REGISTRY } from "#/manifests/manifest-sources";
import {
  flattenRecommendedRailGroups,
  getRecommendedRailGroups,
  isCatalogAutomationAdded,
  isConversationLaunchAutomation,
  normalizeAutomationKey,
} from "#/utils/recommended-automation-rail";

const prReviewer = AUTOMATION_CATALOG.find(
  (entry) => entry.id === "github-pr-reviewer",
)!;
const slackStandup = AUTOMATION_CATALOG.find(
  (entry) => entry.id === "slack-standup-digest",
)!;
const upstreamFork = AUTOMATION_CATALOG.find(
  (entry) => entry.id === "upstream-fork-sync",
)!;

describe("recommended automation rail", () => {
  it("normalizes catalog ids and human titles to the same key", () => {
    expect(normalizeAutomationKey("GitHub Code Review Agent")).toBe(
      "github-code-review-agent",
    );
    expect(normalizeAutomationKey("  slack-standup-digest  ")).toBe(
      "slack-standup-digest",
    );
  });

  it("treats an installed automation as added when the name matches id, title, or skill", () => {
    expect(
      isCatalogAutomationAdded(prReviewer, [
        { name: "GitHub Code Review Agent" },
      ]),
    ).toBe(true);
    expect(
      isCatalogAutomationAdded(prReviewer, [{ name: "github-pr-reviewer" }]),
    ).toBe(true);
    expect(
      isCatalogAutomationAdded(slackStandup, [{ name: "Daily digest" }]),
    ).toBe(false);
  });

  it("keeps proven workflows first and drops ones that have already been added", () => {
    const groups = getRecommendedRailGroups([
      { name: "GitHub Code Review Agent" },
    ]);

    expect(groups.proven.map((entry) => entry.id)).toEqual([
      "github-issue-to-pr",
      "slack-channel-monitor",
      "github-agents-md-maintainer",
      "news-digest",
    ]);
  });

  it("appends other useful automations that open in a new conversation", () => {
    const groups = getRecommendedRailGroups([]);
    const conversationIds = groups.conversation.map((entry) => entry.id);

    expect(groups.proven.map((entry) => entry.id)).toEqual([
      "github-pr-reviewer",
      "github-issue-to-pr",
      "slack-channel-monitor",
      "github-agents-md-maintainer",
      "news-digest",
    ]);
    expect(conversationIds).toEqual([
      "slack-standup-digest",
      "linear-triage-assistant",
      "linear-issue-to-github-pr",
      "linear-issue-to-gitlab-mr",
      "linear-issue-to-bitbucket-pr",
      "jira-issue-to-pr",
      "jira-issue-to-gitlab-mr",
      "research-brief-writer",
      "jira-issue-to-bitbucket-pr",
    ]);
    expect(conversationIds).not.toContain("upstream-fork-sync");
    expect(conversationIds).not.toContain("incident-retrospective-drafter");
    // Beta as of extensions 0.18.0, and set up by a host form rather than a
    // conversation, so it is in neither group.
    expect(conversationIds).not.toContain("github-repo-monitor");
    // Same for `qa-changes` (extensions 0.19.0): host form, so neither group.
    expect(conversationIds).not.toContain("qa-changes");
    expect(isConversationLaunchAutomation(slackStandup)).toBe(true);
    expect(isConversationLaunchAutomation(upstreamFork)).toBe(false);
    expect(SETUP_REGISTRY.findById(upstreamFork.id)).not.toBeNull();
  });

  it("returns an empty rail when every recommended automation has been added", () => {
    const groups = getRecommendedRailGroups([
      { name: "GitHub Code Review Agent" },
      { name: "GitHub Issue to PR Agent" },
      { name: "Slack channel monitor" },
      { name: "AGENTS.md Maintainer" },
      { name: "Daily news digest" },
      { name: "Slack standup digest" },
      { name: "Linear issue triage assistant" },
      { name: "Jira issue to GitHub PR" },
      { name: "Research brief writer" },
      { name: "Linear issue to GitHub PR" },
      { name: "Linear issue to GitLab MR" },
      { name: "Linear issue to Bitbucket PR" },
      { name: "Jira issue to GitLab MR" },
      { name: "Jira issue to Bitbucket PR" },
    ]);

    expect(flattenRecommendedRailGroups(groups)).toEqual([]);
  });
});
