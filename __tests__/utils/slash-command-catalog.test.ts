import { describe, expect, it } from "vitest";
import { buildSlashCommandCatalog } from "#/utils/slash-command-catalog";
import type { SkillInfo } from "#/types/settings";
import { CLI_HELP_COMMAND_ORDER } from "#/utils/constants";

const skill = (
  name: string,
  triggers: string[],
  type: SkillInfo["type"] = "agentskills",
): SkillInfo => ({
  name,
  triggers,
  type,
  source: "project",
  content: `${name} description`,
});

describe("buildSlashCommandCatalog", () => {
  it("matches CLI help order, then Canvas commands, then sorted skills", () => {
    const commands = buildSlashCommandCatalog({
      isCloud: false,
      hasConversation: true,
      agentKind: "openhands",
      supportsManualCondensation: true,
      skills: [
        skill("duplicate", ["/new"]),
        skill("release", ["/release", "/ship"]),
        skill("derived", []),
        skill("knowledge", [], "knowledge"),
      ],
    }).map((item) => item.command);

    expect(commands.slice(0, CLI_HELP_COMMAND_ORDER.length)).toEqual(
      CLI_HELP_COMMAND_ORDER,
    );
    expect(
      commands.slice(
        CLI_HELP_COMMAND_ORDER.length,
        CLI_HELP_COMMAND_ORDER.length + 4,
      ),
    ).toEqual(["/fork", "/btw", "/model", "/goal"]);
    expect(commands.slice(CLI_HELP_COMMAND_ORDER.length + 4)).toEqual([
      "/derived",
      "/release",
      "/ship",
    ]);
  });

  it("applies conversation and backend availability", () => {
    const localHome = buildSlashCommandCatalog({
      isCloud: false,
      hasConversation: false,
    }).map((item) => item.command);
    const cloudConversation = buildSlashCommandCatalog({
      isCloud: true,
      hasConversation: true,
    }).map((item) => item.command);

    expect(localHome).toEqual([
      "/help",
      "/history",
      "/settings",
      "/skills",
      "/feedback",
      "/model",
    ]);
    expect(cloudConversation).toContain("/new");
    expect(cloudConversation).toContain("/confirm");
    expect(cloudConversation).not.toContain("/condense");
    expect(cloudConversation).not.toContain("/fork");
  });

  it("keeps /confirm out of local ACP conversations", () => {
    const commands = buildSlashCommandCatalog({
      isCloud: false,
      hasConversation: true,
      agentKind: "acp",
    }).map((item) => item.command);

    expect(commands).not.toContain("/confirm");
    expect(commands).not.toContain("/condense");
    expect(commands).not.toContain("/fork");
  });

  it("reserves unavailable built-in strings from dynamic skills", () => {
    const commands = buildSlashCommandCatalog({
      isCloud: false,
      hasConversation: true,
      agentKind: "acp",
      skills: [
        skill("condense", ["/condense"]),
        skill("confirm", ["/confirm"]),
        skill("fork", ["/fork"]),
        skill("custom-one", ["/custom", "/custom"]),
        skill("custom-two", ["/custom"]),
      ],
    }).map((item) => item.command);

    expect(commands).not.toContain("/condense");
    expect(commands).not.toContain("/confirm");
    expect(commands).not.toContain("/fork");
    expect(commands.filter((command) => command === "/custom")).toHaveLength(1);
  });

  it("only exposes local /condense for a compatible running condenser", () => {
    const unsupported = buildSlashCommandCatalog({
      isCloud: false,
      hasConversation: true,
      agentKind: "openhands",
      supportsManualCondensation: false,
    }).map((item) => item.command);
    const supported = buildSlashCommandCatalog({
      isCloud: false,
      hasConversation: true,
      agentKind: "openhands",
      supportsManualCondensation: true,
    }).map((item) => item.command);

    expect(unsupported).not.toContain("/condense");
    expect(supported).toContain("/condense");
  });
});
