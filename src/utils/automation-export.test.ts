import { describe, expect, it } from "vitest";
import type { Automation } from "#/types/automation";
import {
  AutomationFileValidationError,
  parseAutomationFile,
  serializeAutomation,
} from "./automation-export";

const cronAutomation: Automation = {
  id: "automation-1",
  name: "Daily review",
  trigger: {
    type: "cron",
    schedule: "0 9 * * *",
    schedule_human: "Daily at 09:00",
  },
  enabled: true,
  repository: "openhands/agent-canvas",
  branch: "main",
  model: "fast",
  prompt: "Review open pull requests.",
  plugins: ["github:openhands/extensions"],
  notification: "Post a summary",
  timezone: "America/Los_Angeles",
  created_at: "2026-07-01T00:00:00Z",
  updated_at: "2026-07-02T00:00:00Z",
  last_triggered_at: "2026-07-03T00:00:00Z",
};

describe("automation export files", () => {
  it("round-trips a cron automation without server-assigned fields", () => {
    const exported = serializeAutomation(cronAutomation);

    expect(exported).toEqual({
      version: 1,
      kind: "automation",
      spec: {
        name: cronAutomation.name,
        trigger: cronAutomation.trigger,
        enabled: true,
        repository: cronAutomation.repository,
        model: cronAutomation.model,
        prompt: cronAutomation.prompt,
        branch: cronAutomation.branch,
        plugins: cronAutomation.plugins,
        notification: cronAutomation.notification,
        timezone: cronAutomation.timezone,
      },
    });
    expect(parseAutomationFile(exported)).toEqual(exported.spec);
    expect(exported.spec).not.toHaveProperty("id");
    expect(exported.spec).not.toHaveProperty("created_at");
    expect(exported.spec).not.toHaveProperty("updated_at");
    expect(exported.spec).not.toHaveProperty("last_triggered_at");
  });

  it("round-trips event trigger fields", () => {
    const exported = serializeAutomation({
      ...cronAutomation,
      trigger: {
        type: "event",
        source: "github",
        on: ["pull_request.opened", "pull_request.synchronize"],
        filter: "repository.full_name == 'openhands/agent-canvas'",
      },
      timezone: undefined,
    });

    expect(parseAutomationFile(exported).trigger).toEqual(
      exported.spec.trigger,
    );
  });

  it("preserves a positive timeout through an export/import round-trip", () => {
    const withTimeout: Automation = { ...cronAutomation, timeout: 900 };

    const exported = serializeAutomation(withTimeout);

    expect(parseAutomationFile(exported).timeout).toBe(900);
  });

  it("reports every malformed field with its path", () => {
    const malformed = {
      version: 2,
      kind: "workflow",
      spec: {
        name: "",
        prompt: null,
        enabled: "yes",
        trigger: { type: "event", source: 42, on: [] },
        plugins: ["github:openhands/extensions", 42],
        model: false,
        timeout: -5,
      },
    };

    expect(() => parseAutomationFile(malformed)).toThrow(
      AutomationFileValidationError,
    );

    try {
      parseAutomationFile(malformed);
    } catch (error) {
      expect(error).toBeInstanceOf(AutomationFileValidationError);
      expect((error as AutomationFileValidationError).issues).toEqual(
        expect.arrayContaining([
          "version: expected 1",
          'kind: expected "automation"',
          "spec.name: expected a non-empty string",
          "spec.prompt: expected a non-empty string",
          "spec.enabled: expected a boolean",
          "spec.trigger.source: expected a string",
          "spec.trigger.source: required for an event trigger",
          "spec.trigger.on: expected a string or a non-empty array of strings",
          "spec.trigger.on: required for an event trigger",
          "spec.plugins: expected an array of non-empty strings",
          "spec.model: expected a non-empty string or null",
          "spec.timeout: expected a positive integer",
        ]),
      );
    }
  });

  it("rejects non-object files and missing specs", () => {
    expect(() => parseAutomationFile(null)).toThrow(
      "file: expected a JSON object",
    );
    expect(() =>
      parseAutomationFile({ version: 1, kind: "automation" }),
    ).toThrow("spec: expected an object");
  });
});
