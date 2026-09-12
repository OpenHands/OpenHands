import { afterEach, describe, expect, it } from "vitest";

import {
  externalSkillToAgentSkill,
  externalSkillToSkillInfo,
  findInvokedExternalSkill,
  getExternalSkillEntries,
  type ExternalSkillEntry,
} from "#/utils/external-skills";

const WINDOW_KEY = "__AGENT_CANVAS_EXTERNAL_SKILLS__";

function setWindowManifest(value: unknown) {
  (window as unknown as Record<string, unknown>)[WINDOW_KEY] = value;
}

afterEach(() => {
  delete (window as unknown as Record<string, unknown>)[WINDOW_KEY];
});

const ENTRY: ExternalSkillEntry = {
  name: "acme-deploy",
  description: "Deploy the Acme way.",
  triggers: ["/deploy"],
  content: "# Deploy\n\nShip it.",
  license: "MIT",
  compatibility: null,
  version: "1.0.0",
  allowed_tools: ["Bash"],
  path: "/home/u/.openhands/agent-canvas/external-skills/cache/acme-1234abcd/acme-deploy/SKILL.md",
  source: "github.com/acmecorp/skills",
  source_id: "acme-1234abcd",
};

describe("getExternalSkillEntries", () => {
  it("returns [] when the launcher injected nothing", () => {
    expect(getExternalSkillEntries()).toEqual([]);
    setWindowManifest("");
    expect(getExternalSkillEntries()).toEqual([]);
  });

  it("parses the injected JSON-string manifest", () => {
    setWindowManifest(
      JSON.stringify({ version: 1, sources: [], skills: [ENTRY] }),
    );
    expect(getExternalSkillEntries()).toEqual([ENTRY]);
  });

  it("tolerates an already-parsed manifest object", () => {
    setWindowManifest({ version: 1, sources: [], skills: [ENTRY] });
    expect(getExternalSkillEntries()).toEqual([ENTRY]);
  });

  it("returns [] for malformed JSON or a malformed manifest", () => {
    setWindowManifest("{not json");
    expect(getExternalSkillEntries()).toEqual([]);
    setWindowManifest(JSON.stringify({ version: 1 }));
    expect(getExternalSkillEntries()).toEqual([]);
  });

  it("drops entries that lack the required identity fields", () => {
    setWindowManifest({
      skills: [
        ENTRY,
        { ...ENTRY, name: "no-source-id", source_id: undefined },
        { ...ENTRY, name: "no-content", content: undefined },
        { ...ENTRY, name: "bad-triggers", triggers: "not-a-list" },
      ],
    });
    expect(getExternalSkillEntries()).toEqual([ENTRY]);
  });
});

describe("externalSkillToSkillInfo", () => {
  it("maps a manifest entry to a SkillInfo card", () => {
    expect(externalSkillToSkillInfo(ENTRY)).toEqual({
      name: "acme-deploy",
      type: "agentskills",
      source: "github.com/acmecorp/skills",
      source_id: "acme-1234abcd",
      description: "Deploy the Acme way.",
      triggers: ["/deploy"],
      content: "# Deploy\n\nShip it.",
      license: "MIT",
      compatibility: null,
      version: "1.0.0",
      allowed_tools: ["Bash"],
      is_agentskills_format: true,
    });
  });
});

describe("findInvokedExternalSkill", () => {
  it("resolves /<name> and declared slash triggers, leading token only", () => {
    setWindowManifest({ skills: [ENTRY] });
    expect(findInvokedExternalSkill("/acme-deploy now")).toEqual(ENTRY);
    expect(findInvokedExternalSkill("/deploy")).toEqual(ENTRY);
    expect(findInvokedExternalSkill("please /deploy")).toBeUndefined();
    expect(findInvokedExternalSkill("no command")).toBeUndefined();
    expect(findInvokedExternalSkill(undefined)).toBeUndefined();
  });

  it("returns undefined with no external skills or no slash command", () => {
    expect(findInvokedExternalSkill("/acme-deploy")).toBeUndefined();
    setWindowManifest({ skills: [ENTRY] });
    expect(findInvokedExternalSkill("/unrelated")).toBeUndefined();
  });
});

describe("externalSkillToAgentSkill", () => {
  it("builds the agent_context.skills payload", () => {
    expect(externalSkillToAgentSkill(ENTRY)).toEqual({
      name: "acme-deploy",
      content: "# Deploy\n\nShip it.",
      trigger: { type: "keyword", keywords: ["/deploy"] },
      // The absolute SKILL.md path lets the agent-server resolve relative
      // resources (scripts/, references/) exactly like bundled skills.
      source: ENTRY.path,
      description: "Deploy the Acme way.",
      is_agentskills_format: true,
      license: "MIT",
    });
  });

  it("emits a null trigger and qualified fallback source when absent", () => {
    const minimal: ExternalSkillEntry = {
      name: "bare",
      content: "Body.",
      source: "github.com/acmecorp/skills",
      source_id: "acme-1234abcd",
    };
    expect(externalSkillToAgentSkill(minimal)).toEqual({
      name: "bare",
      content: "Body.",
      trigger: null,
      source: "external:acme-1234abcd",
      description: null,
      is_agentskills_format: true,
    });
  });
});
