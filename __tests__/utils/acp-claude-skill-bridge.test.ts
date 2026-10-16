import { describe, expect, it, vi } from "vitest";
import {
  DEFAULT_ENABLED_SKILL_NAMES,
  SKILLS_CATALOG,
} from "@openhands/extensions/skills";
import {
  AGENT_LAUNCH_SUFFIX_APPEND_MAX_LENGTH,
  CANVAS_ENABLED_SKILLS_TAG,
  CANVAS_SKILLS_TRUNCATED_MARKER,
  CLAUDE_CODE_ACP_SERVER,
  buildClaudeAcpSkillSuffixAppend,
  fenceSafeTruncate,
  isClaudeCodeAcpAgent,
  packClaudeAcpSkillSuffix,
  prioritizeAutomationContent,
  selectClaudeAcpCatalogSkills,
} from "#/utils/acp-claude-skill-bridge";

const AUTOMATION = "openhands-automation";

describe("isClaudeCodeAcpAgent", () => {
  it("matches the Claude Code registry key", () => {
    expect(
      isClaudeCodeAcpAgent({
        agentKind: "acp",
        acpServer: CLAUDE_CODE_ACP_SERVER,
      }),
    ).toBe(true);
  });

  it("matches a Claude ACP adapter command without the registry key", () => {
    expect(
      isClaudeCodeAcpAgent({
        agentKind: "acp",
        acpCommand: [
          "npx",
          "-y",
          "@agentclientprotocol/claude-agent-acp@0.63.0",
        ],
      }),
    ).toBe(true);
  });

  it("rejects OpenHands and other ACP harnesses", () => {
    expect(
      isClaudeCodeAcpAgent({
        agentKind: "openhands",
        acpServer: "claude-code",
      }),
    ).toBe(false);
    expect(isClaudeCodeAcpAgent({ agentKind: "acp", acpServer: "codex" })).toBe(
      false,
    );
    expect(
      isClaudeCodeAcpAgent({
        agentKind: "acp",
        acpServer: "custom",
        acpCommand: "python3 mock-acp-server.py",
      }),
    ).toBe(false);
  });
});

describe("selectClaudeAcpCatalogSkills", () => {
  it("includes default-enabled catalog skills and skips optional ones", () => {
    const names = selectClaudeAcpCatalogSkills({}).map((skill) => skill.name);
    expect(names).toContain(AUTOMATION);
    expect(names).not.toContain("add-javadoc");
  });

  it("lets disabled_skills veto an allow-listed catalog skill", () => {
    const names = selectClaudeAcpCatalogSkills({
      enabledSkills: [AUTOMATION, "add-skill"],
      disabledSkills: [AUTOMATION],
    }).map((skill) => skill.name);
    expect(names).not.toContain(AUTOMATION);
    expect(names).toContain("add-skill");
  });

  it("includes a slash-invoked catalog skill even when it is off", () => {
    const names = selectClaudeAcpCatalogSkills(
      { enabledSkills: ["add-skill"] },
      "slack-standup-digest",
    ).map((skill) => skill.name);
    expect(names[0]).toBe("slack-standup-digest");
    expect(names).toContain("add-skill");
  });
});

describe("packClaudeAcpSkillSuffix", () => {
  it("wraps packed skills and stays within the launch-addition cap", () => {
    const suffix = packClaudeAcpSkillSuffix(
      SKILLS_CATALOG.filter((entry) => entry.name === AUTOMATION),
    );
    expect(suffix).toContain(`<${CANVAS_ENABLED_SKILLS_TAG}>`);
    expect(suffix).toContain("# OpenHands Automations");
    expect(suffix!.length).toBeLessThanOrEqual(
      AGENT_LAUNCH_SUFFIX_APPEND_MAX_LENGTH,
    );
  });

  it("reserves per-skill budget so a small skill is not silently evicted", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const suffix = packClaudeAcpSkillSuffix(
      [
        { name: "tiny", content: "tiny body" },
        { name: AUTOMATION, content: "A".repeat(20_000) },
      ],
      4_000,
    );
    expect(suffix).toContain(`## ${AUTOMATION}`);
    expect(suffix).toContain("## tiny");
    expect(suffix).toContain(CANVAS_SKILLS_TRUNCATED_MARKER);
    expect(suffix).toMatch(/truncated="[^"]*openhands-automation/);
    warn.mockRestore();
  });

  it("returns undefined when nothing is selected", () => {
    expect(packClaudeAcpSkillSuffix([])).toBeUndefined();
  });
});

describe("buildClaudeAcpSkillSuffixAppend", () => {
  it("projects openhands-automation when it is enabled", () => {
    const suffix = buildClaudeAcpSkillSuffixAppend({
      enablement: { enabledSkills: [AUTOMATION] },
    });
    expect(suffix).toContain("# OpenHands Automations");
    expect(suffix!.length).toBeLessThanOrEqual(
      AGENT_LAUNCH_SUFFIX_APPEND_MAX_LENGTH,
    );
  });

  it("omits openhands-automation when it is disabled", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const suffix = buildClaudeAcpSkillSuffixAppend({
      enablement: {
        enabledSkills: ["add-skill"],
        disabledSkills: [AUTOMATION],
      },
    });
    expect(suffix ?? "").not.toContain("# OpenHands Automations");
    warn.mockRestore();
  });

  it("keeps automation schedule/lifecycle sections under the 32KiB default pack", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const suffix = buildClaudeAcpSkillSuffixAppend({
      enablement: { enabledSkills: [...DEFAULT_ENABLED_SKILL_NAMES] },
    });

    expect(suffix).toBeDefined();
    expect(suffix!.length).toBeLessThanOrEqual(
      AGENT_LAUNCH_SUFFIX_APPEND_MAX_LENGTH,
    );
    expect(suffix).toContain(`## ${AUTOMATION}`);
    // #16905 criterion 1 — schedule/lifecycle instructions must survive.
    expect(suffix).toContain("## Managing Automations");
    expect(suffix).toContain("## Run Lifecycle");

    const present = DEFAULT_ENABLED_SKILL_NAMES.filter((name) =>
      suffix!.includes(`## ${name}`),
    );
    expect(present).toContain(AUTOMATION);
    // Other defaults may still appear when leftover budget allows, but the
    // pack must not require every heading (that starved automation before).
    expect(present.length).toBeGreaterThan(1);

    expect(suffix).toContain(CANVAS_SKILLS_TRUNCATED_MARKER);
    expect(suffix).toMatch(/truncated="/);

    const markerIdx = suffix!.indexOf(CANVAS_SKILLS_TRUNCATED_MARKER);
    const beforeMarker = suffix!.slice(0, markerIdx);
    const fenceCount = (beforeMarker.match(/```/g) ?? []).length;
    expect(fenceCount % 2).toBe(0);
    warn.mockRestore();
  });
});

describe("fenceSafeTruncate", () => {
  it("closes an odd fence so markers stay outside code", () => {
    const src = "before\n```\ncode line\nmore";
    const cut = fenceSafeTruncate(src, 24);
    expect((cut.match(/```/g) ?? []).length % 2).toBe(0);
  });
});

describe("prioritizeAutomationContent", () => {
  it("keeps Managing Automations over Architecture when budget is tight", () => {
    const full = SKILLS_CATALOG.find((e) => e.name === AUTOMATION)!.content;
    const { content, truncated } = prioritizeAutomationContent(full, 8_000);
    expect(truncated).toBe(true);
    expect(content).toContain("## Managing Automations");
    expect(content).toContain("## Run Lifecycle");
    expect(content).not.toContain("## Architecture");
  });
});
