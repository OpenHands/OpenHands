import { describe, expect, it } from "vitest";
import type { PluginSpecPreflightResponse } from "#/api/automation-service/automation-service.api";
import type { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";
import {
  applySchemaDefaults,
  buildCronTrigger,
  configSchemaToFormFields,
  pluginSpecTemplateId,
} from "./plugin-launch-preflight";

function makePreflight(
  manifest: PluginSpecPreflightResponse["manifest"],
  suggested_trigger?: PluginSpecPreflightResponse["suggested_trigger"],
): PluginSpecPreflightResponse {
  return { manifest, suggested_trigger: suggested_trigger ?? null };
}

describe("configSchemaToFormFields", () => {
  it("maps schema properties to typed fields", () => {
    const fields = configSchemaToFormFields(
      makePreflight({
        name: "p",
        config_schema: {
          type: "object",
          required: ["repo"],
          properties: {
            repo: { type: "string", title: "Repository" },
            count: { type: "integer" },
            verbose: { type: "boolean", description: "Log more" },
            style: { type: "string", enum: ["compact", "full"] },
          },
        },
      }),
    );

    expect(fields).toHaveLength(4);
    expect(fields[0]).toMatchObject({
      key: "repo",
      type: "string",
      label: "Repository",
      required: true,
    });
    expect(fields[1]).toMatchObject({ key: "count", type: "number" });
    expect(fields[2]).toMatchObject({
      key: "verbose",
      type: "boolean",
      helpText: "Log more",
      required: false,
    });
    expect(fields[3]).toMatchObject({
      key: "style",
      options: ["compact", "full"],
    });
  });

  it("returns no fields without a config schema", () => {
    expect(configSchemaToFormFields(makePreflight({ name: "p" }))).toEqual([]);
  });

  it("falls back to the key when the property has no title", () => {
    const fields = configSchemaToFormFields(
      makePreflight({
        name: "p",
        config_schema: {
          type: "object",
          properties: { repo: { type: "string" } },
        },
      }),
    );
    expect(fields[0]?.label).toBe("repo");
  });

  it("maps format: textarea and ui:widget: textarea to the textarea widget", () => {
    const fields = configSchemaToFormFields(
      makePreflight({
        name: "p",
        config_schema: {
          type: "object",
          properties: {
            feeds: { type: "string", format: "textarea" },
            topics: { type: "string", "ui:widget": "textarea" },
            name: { type: "string" },
          },
        },
      }),
    );
    expect(fields[0]?.widget).toBe("textarea");
    expect(fields[1]?.widget).toBe("textarea");
    expect(fields[2]?.widget).toBeUndefined();
  });
});

describe("applySchemaDefaults", () => {
  const fields = configSchemaToFormFields(
    makePreflight({
      name: "p",
      config_schema: {
        type: "object",
        properties: {
          repo: { type: "string", default: "org/repo" },
          count: { type: "number" },
          verbose: { type: "boolean" },
        },
      },
    }),
  );

  it("fills schema defaults for keys the URL omitted", () => {
    const merged = applySchemaDefaults({ count: 7 }, fields);
    expect(merged).toEqual({ count: 7, repo: "org/repo", verbose: false });
  });

  it("never overwrites URL-encoded values", () => {
    const merged = applySchemaDefaults({ repo: "other/repo" }, fields);
    expect(merged.repo).toBe("other/repo");
  });
});

describe("pluginSpecTemplateId", () => {
  it("includes source, repo_path, and ref", () => {
    const plugin: PluginSpec = {
      source: "github:org/recipe",
      ref: "v2",
      repo_path: "plugins/recipe",
      parameters: null,
    };
    expect(pluginSpecTemplateId(plugin)).toBe(
      "plugin-spec:github:org/recipe/plugins/recipe/v2",
    );
  });

  it("is stable for plugins without ref or path", () => {
    const plugin: PluginSpec = {
      source: "github:org/recipe",
      ref: null,
      repo_path: null,
      parameters: null,
    };
    expect(pluginSpecTemplateId(plugin)).toBe("plugin-spec:github:org/recipe");
  });
});

describe("buildCronTrigger", () => {
  it("returns the (possibly edited) cron trigger", () => {
    expect(
      buildCronTrigger(
        { type: "cron", schedule: "0 9 * * *" },
        "*/5 * * * *",
        "UTC",
      ),
    ).toEqual({ type: "cron", schedule: "*/5 * * * *", timezone: "UTC" });
  });

  it("returns null without a cron suggestion", () => {
    expect(buildCronTrigger(null, "x", "UTC")).toBeNull();
    expect(buildCronTrigger({ type: "event" }, "x", "UTC")).toBeNull();
  });
});
