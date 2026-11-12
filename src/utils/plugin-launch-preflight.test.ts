import { describe, expect, it } from "vitest";
import type { PluginSpecPreflightResponse } from "#/api/automation-service/automation-service.api";
import type { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";
import {
  applySchemaDefaults,
  buildCronTrigger,
  configSchemaToFormFields,
  pluginSpecTemplateId,
  stripEmptyConfigValues,
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
    expect(fields[1]).toMatchObject({ key: "count", type: "integer" });
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
    // Only real defaults are filled; count (URL-encoded) and verbose (no
    // default) are left as-is.
    expect(merged).toEqual({ count: 7, repo: "org/repo" });
  });

  it("never overwrites URL-encoded values", () => {
    const merged = applySchemaDefaults({ repo: "other/repo" }, fields);
    expect(merged.repo).toBe("other/repo");
  });

  it("does not pre-fill empty values for fields without defaults", () => {
    // verbose has no default; it must not be pre-filled with false, which
    // would install a broken automation (boolean false sent as a real value)
    // and mask a required-check failure on the backend.
    const merged = applySchemaDefaults({}, fields);
    expect(merged).toEqual({ repo: "org/repo" });
    expect("verbose" in merged).toBe(false);
  });
});

describe("stripEmptyConfigValues", () => {
  it("removes undefined, null, and empty-string values", () => {
    const cleaned = stripEmptyConfigValues({
      feeds: "https://example.com/rss",
      topics: "",
      max_items: undefined,
      verbose: null,
      kept: 0,
    });
    expect(cleaned).toEqual({
      feeds: "https://example.com/rss",
      kept: 0,
    });
  });
});

describe("pluginSpecTemplateId", () => {
  it("includes source, repo_path, and ref as positional slots", () => {
    const plugin: PluginSpec = {
      source: "github:org/recipe",
      ref: "v2",
      repo_path: "plugins/recipe",
      parameters: null,
    };
    expect(pluginSpecTemplateId(plugin)).toBe(
      "plugin-spec:github:org/recipe|plugins/recipe|v2",
    );
  });

  it("is stable for plugins without ref or path", () => {
    const plugin: PluginSpec = {
      source: "github:org/recipe",
      ref: null,
      repo_path: null,
      parameters: null,
    };
    expect(pluginSpecTemplateId(plugin)).toBe(
      "plugin-spec:github:org/recipe||",
    );
  });

  it("does not collide when repo_path contains a value that matches a ref", () => {
    // With a `/`-joined id, `github:org/repo|main` and `github:org/repo/main`
    // with no ref would both produce `plugin-spec:github:org/repo/main`.
    // Positional slots keep them distinct.
    const a: PluginSpec = {
      source: "github:org/repo",
      ref: "main",
      repo_path: null,
      parameters: null,
    };
    const b: PluginSpec = {
      source: "github:org/repo",
      ref: null,
      repo_path: "main",
      parameters: null,
    };
    expect(pluginSpecTemplateId(a)).not.toBe(pluginSpecTemplateId(b));
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
