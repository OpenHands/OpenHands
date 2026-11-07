import { describe, expect, it } from "vitest";

import {
  AUTOMATION_FORM_UPDATE_ACTION_KIND,
  AUTOMATION_FORM_UPDATE_TOOL_NAME,
  buildAutomationFormUpdateClientTool,
} from "#/api/automation-form-client-tool";
import type { DeploymentCapabilities } from "#/manifests/types";

// The tool as sent when the deployment's capabilities are not known.
const defaultTool = buildAutomationFormUpdateClientTool();

// An Enterprise instance with Jira and Bitbucket Data Center built in.
const DATA_CENTER_CAPABILITIES: DeploymentCapabilities = {
  ready: true,
  triggerKinds: ["cron", "event"],
  eventSources: ["bitbucket_data_center", "github", "jira_dc"],
  eventTypes: ["issue_comment.created", "pull_request.opened"],
  eventTypesBySource: {
    bitbucket_data_center: [
      "pr:from_ref_updated",
      "pr:opened",
      "repo:refs_changed",
    ],
    github: ["issue_comment.created", "pull_request.opened"],
    jira_dc: ["comment_created", "jira:issue_created", "jira:issue_updated"],
  },
  triggers: {},
  features: ["webhookDelivery"],
};

describe("automation_form_update client tool", () => {
  it("defines the automation form update client tool contract", () => {
    expect(AUTOMATION_FORM_UPDATE_TOOL_NAME).toBe("automation_form_update");
    expect(AUTOMATION_FORM_UPDATE_ACTION_KIND).toBe(
      "ClientAction_automation_form_update",
    );
    expect(defaultTool).toMatchObject({
      name: AUTOMATION_FORM_UPDATE_TOOL_NAME,
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          fields: {
            type: "object",
            additionalProperties: false,
            properties: {
              kind: { enum: ["prompt", "plugin", "custom"] },
              triggerKind: { enum: ["cron", "event"] },
              frequency: {
                enum: [
                  "once",
                  "hourly",
                  "daily",
                  "weekdays",
                  "weekly",
                  "custom",
                ],
              },
              customCode: { type: "string" },
              setupScript: { type: "string" },
            },
          },
          overwrite_user_edits: { type: "boolean" },
        },
        required: ["fields"],
      },
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: false,
        openWorldHint: false,
      },
    });
  });

  it("describes repositories on any of the instance's git providers", () => {
    const repositoryLine = defaultTool.description
      .split("\n")
      .find((line) => line.startsWith("* repository:"));

    expect(repositoryLine).toMatch(/Bitbucket Cloud or Data Center/);
    expect(repositoryLine).toMatch(/owner\/repo for GitHub/);
  });

  it("names the deployment's event sources with their keys and a tested filter", () => {
    const { description } = buildAutomationFormUpdateClientTool(
      DATA_CENTER_CAPABILITIES,
    );
    const sourceLine = (source: string) =>
      description.split("\n").find((line) => line.startsWith(`* ${source}`)) ??
      "";

    expect(sourceLine("bitbucket_data_center")).toMatch(
      /event keys pr:from_ref_updated, pr:opened, repo:refs_changed\./,
    );
    expect(sourceLine("bitbucket_data_center")).toMatch(
      /names the event in "eventKey"/,
    );
    expect(sourceLine("bitbucket_data_center")).toContain(
      "eventKey \"pr:*\" with eventFilter (eventKey == 'pr:opened' || eventKey == 'pr:from_ref_updated') && pullRequest.draft != `true` && join('/', [pullRequest.toRef.repository.project.key, pullRequest.toRef.repository.slug]) == 'ENG/payments'",
    );
    expect(sourceLine("bitbucket_data_center")).toMatch(
      /Replace ENG\/payments with the project key and repository slug/,
    );

    expect(sourceLine("jira_dc")).toMatch(
      /event keys comment_created, jira:issue_created, jira:issue_updated\./,
    );
    expect(sourceLine("jira_dc")).toMatch(/names the event in "webhookEvent"/);
    expect(sourceLine("jira_dc")).toContain(
      "eventKey \"jira:issue_*\" with eventFilter issue.fields.project.key == 'ENG' && ((webhookEvent == 'jira:issue_created' && contains(issue.fields.labels || `[]`, 'openhands')) || (webhookEvent == 'jira:issue_updated' && length(changelog.items[?field == 'labels'",
    );
    expect(sourceLine("jira_dc")).toMatch(
      /Replace ENG with the Jira project key and openhands with the label that starts a run/,
    );

    // A source Canvas has no guide for still gets its keys.
    expect(sourceLine("github")).toBe(
      "* github: event keys issue_comment.created, pull_request.opened.",
    );
    expect(description).not.toContain("azure_devops");
  });

  it("lists GitHub's keys from an older service that has no per-source map", () => {
    const { description } = buildAutomationFormUpdateClientTool({
      ...DATA_CENTER_CAPABILITIES,
      eventSources: ["azure_devops", "github"],
      eventTypesBySource: undefined,
    });

    expect(description).toContain(
      "* github: event keys issue_comment.created, pull_request.opened.",
    );
    expect(description).toContain(
      "* azure_devops: any event key the source delivers.",
    );
  });

  it("tells the assistant to ask, or to use cron, when sources are unknown or absent", () => {
    expect(buildAutomationFormUpdateClientTool(null).description).toMatch(
      /could not be read\. Ask the user which integration delivers the event/,
    );
    expect(
      buildAutomationFormUpdateClientTool({
        ...DATA_CENTER_CAPABILITIES,
        eventSources: [],
      }).description,
    ).toMatch(/delivers no webhook events.*use triggerKind "cron"/);
  });

  it("keeps the parameters schema identical whatever the deployment reports", () => {
    // The agent-server caches one schema per tool name for the life of the
    // process, so a per-deployment description must not change the schema.
    const withSources = buildAutomationFormUpdateClientTool(
      DATA_CENTER_CAPABILITIES,
    );

    expect(withSources.name).toBe(defaultTool.name);
    expect(withSources.parameters).toEqual(defaultTool.parameters);
    expect(withSources.annotations).toEqual(defaultTool.annotations);
    expect(withSources.description).not.toBe(defaultTool.description);
  });
});
