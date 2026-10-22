import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import { resolveFeaturePolicy } from "../../scripts/feature-policy.mjs";
import { FEATURE_IDS } from "#/feature-policy/feature-policy";

const tempDirs: string[] = [];

function writePolicyFile(contents: string) {
  const dir = mkdtempSync(path.join(tmpdir(), "feature-policy-"));
  tempDirs.push(dir);
  const file = path.join(dir, "policy.json");
  writeFileSync(file, contents);
  return file;
}

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe("resolveFeaturePolicy", () => {
  it("returns null when no policy is configured", () => {
    expect(resolveFeaturePolicy({})).toBeNull();
  });

  it("reads an inline policy and drops duplicate IDs", () => {
    const policy = resolveFeaturePolicy({
      AGENT_CANVAS_FEATURES: JSON.stringify({
        version: 1,
        disabled: ["backends.add", "backends.add"],
      }),
    });

    expect(policy).toEqual({
      version: 1,
      disabled: ["backends.add"],
    });
  });

  it("reads a policy file", () => {
    const file = writePolicyFile(
      JSON.stringify({ version: 1, disabled: ["backends.add"] }),
    );

    expect(resolveFeaturePolicy({ AGENT_CANVAS_FEATURES_FILE: file })).toEqual({
      version: 1,
      disabled: ["backends.add"],
    });
  });

  // An invalid policy must stop startup, never fall back to the full UI.
  it.each([
    {
      name: "both variables set",
      env: {
        AGENT_CANVAS_FEATURES: '{"version":1,"disabled":[]}',
        AGENT_CANVAS_FEATURES_FILE: "/policy.json",
      },
      error: /not both/,
    },
    {
      name: "invalid JSON",
      env: { AGENT_CANVAS_FEATURES: "{version:1" },
      error: /Invalid feature policy JSON/,
    },
    {
      name: "unsupported version",
      env: { AGENT_CANVAS_FEATURES: '{"version":2,"disabled":[]}' },
      error: /Unsupported feature policy version/,
    },
    {
      name: "non-array disabled",
      env: {
        AGENT_CANVAS_FEATURES: '{"version":1,"disabled":"backends.add"}',
      },
      error: /must be an array/,
    },
    {
      name: "unknown feature ID",
      env: {
        AGENT_CANVAS_FEATURES: '{"version":1,"disabled":["backend.add"]}',
      },
      error: /Unknown feature IDs .*: backend\.add/,
    },
    {
      name: "unreadable file",
      env: { AGENT_CANVAS_FEATURES_FILE: "/nonexistent/policy.json" },
      error: /Cannot read feature policy file/,
    },
  ])("throws on $name", ({ env, error }) => {
    expect(() => resolveFeaturePolicy(env)).toThrow(error);
  });
});

describe("feature catalog", () => {
  // The frontend's IDs and the launcher's catalog must agree, or an operator
  // could not disable a feature the UI gates.
  it("lists the same IDs as the frontend", () => {
    const catalog = JSON.parse(
      readFileSync(path.join(__dirname, "../../config/features.json"), "utf8"),
    ) as { features: { id: string }[] };

    expect(catalog.features.map((feature) => feature.id)).toEqual([
      ...FEATURE_IDS,
    ]);
  });
});
