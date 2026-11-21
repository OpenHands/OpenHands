import { describe, expect, it } from "vitest";
import type { Automation } from "#/types/automation";
import {
  AutomationFileValidationError,
  parseAutomationFile,
  serializeAutomation,
  serializeScriptAutomation,
} from "./automation-export";
import { withPresetSources } from "./automation-preset-sources";
import { base64ToBytes, bytesToBase64 } from "./base64";

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

const scriptAutomation: Automation = {
  id: "automation-2",
  name: "Socials monitor",
  trigger: { type: "cron", schedule: "0,30 9-17 * * 1-5", timezone: "UTC" },
  enabled: true,
  prompt: null,
  entrypoint: "python3 main.py",
  setup_script_path: "setup.sh",
  tarball_path: "oh-internal://uploads/socials-monitor",
  model: "fast",
  created_at: "2026-10-05T00:00:00Z",
  updated_at: "2026-10-06T00:00:00Z",
};

const bundle = new Uint8Array([0x1f, 0x8b, 8, 0, 1, 2, 3, 250, 255, 0]);

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

  it("exports the repository, ref and plugins read from preset_metadata", () => {
    const exported = serializeAutomation(
      withPresetSources({
        ...cronAutomation,
        repository: undefined,
        branch: undefined,
        plugins: undefined,
        preset_metadata: {
          repos: [
            { url: "https://github.com/qa-example/qa-repo", ref: "main" },
          ],
          plugins: [{ source: "github:qa-example/qa-plugin" }],
        },
      }),
    );

    expect(exported.spec).toMatchObject({
      repository: "https://github.com/qa-example/qa-repo",
      branch: "main",
      plugins: ["github:qa-example/qa-plugin"],
    });
    expect(exported.spec).not.toHaveProperty("repositories");
    expect(parseAutomationFile(exported)).toEqual(exported.spec);
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

  it("round-trips a script automation with its bundle", () => {
    const exported = serializeScriptAutomation(scriptAutomation, bundle);

    expect(exported.spec).toMatchObject({
      prompt: null,
      script: {
        entrypoint: "python3 main.py",
        setup_script_path: "setup.sh",
        tarball_base64: bytesToBase64(bundle),
      },
    });
    expect(exported.spec).not.toHaveProperty("tarball_path");
    const parsed = parseAutomationFile(JSON.parse(JSON.stringify(exported)));
    expect(parsed).toEqual(exported.spec);
    expect(base64ToBytes(parsed.script!.tarball_base64)).toEqual(bundle);
  });

  it("reports malformed script fields with their paths", () => {
    const exported = serializeScriptAutomation(scriptAutomation, bundle);
    const malformed = {
      ...exported,
      spec: {
        ...exported.spec,
        prompt: "Also a prompt",
        script: { setup_script_path: 7, tarball_base64: "not base64!" },
      },
    };

    expect(() => parseAutomationFile(malformed)).toThrow(
      AutomationFileValidationError,
    );
    try {
      parseAutomationFile(malformed);
    } catch (error) {
      expect((error as AutomationFileValidationError).issues).toEqual(
        expect.arrayContaining([
          "spec.script.entrypoint: expected a non-empty string",
          "spec.script.setup_script_path: expected a string",
          "spec.script.tarball_base64: expected base64",
          "spec.prompt: must be omitted for a script automation",
        ]),
      );
    }
    expect(() =>
      parseAutomationFile({
        ...exported,
        spec: { ...exported.spec, script: 5 },
      }),
    ).toThrow("spec.script: expected an object");
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
