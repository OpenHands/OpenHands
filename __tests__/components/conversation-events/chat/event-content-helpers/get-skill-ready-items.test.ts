import { describe, expect, it } from "vitest";
import {
  getSkillReadyContent,
  getSkillReadyItems,
} from "#/components/conversation-events/chat/event-content-helpers/get-skill-ready-content";
import type { TextContent } from "#/types/agent-server/core/base/common";

const makeTextContent = (text: string): TextContent[] => [
  { type: "text", text },
];

const wrapExtraInfo = (content: string): string =>
  `<EXTRA_INFO>${content}</EXTRA_INFO>`;

describe("getSkillReadyItems", () => {
  it("pairs skills with their EXTRA_INFO blocks by index", () => {
    const skills = ["docker", "gitlab"];
    const extended = makeTextContent(
      `${wrapExtraInfo("Docker guide")}${wrapExtraInfo("GitLab guide")}`,
    );

    const items = getSkillReadyItems(skills, extended);

    expect(items).toEqual([
      { name: "docker", content: "Docker guide" },
      { name: "gitlab", content: "GitLab guide" },
    ]);
  });

  it("returns empty content for skills without matching EXTRA_INFO", () => {
    const skills = ["docker", "gitlab"];
    const extended = makeTextContent(wrapExtraInfo("Docker guide only"));

    const items = getSkillReadyItems(skills, extended);

    expect(items).toEqual([
      { name: "docker", content: "Docker guide only" },
      { name: "gitlab", content: "" },
    ]);
  });

  it("returns unnamed items when no skills but EXTRA_INFO blocks exist", () => {
    const extended = makeTextContent(
      `${wrapExtraInfo("Block A")}${wrapExtraInfo("Block B")}`,
    );

    const items = getSkillReadyItems([], extended);

    expect(items).toEqual([
      { name: "Extended Content 1", content: "Block A" },
      { name: "Extended Content 2", content: "Block B" },
    ]);
  });

  it("returns empty array when no skills and no extended content", () => {
    expect(getSkillReadyItems([], [])).toEqual([]);
  });

  it("skips empty EXTRA_INFO blocks for unnamed items", () => {
    const extended = makeTextContent(
      `${wrapExtraInfo("Content")}${wrapExtraInfo("   ")}`,
    );

    const items = getSkillReadyItems([], extended);

    expect(items).toEqual([{ name: "Extended Content 1", content: "Content" }]);
  });

  it("trims content from EXTRA_INFO blocks", () => {
    const skills = ["docker"];
    const extended = makeTextContent(wrapExtraInfo("  trimmed content  "));

    const items = getSkillReadyItems(skills, extended);

    expect(items[0].content).toBe("trimmed content");
  });
});

describe("skill-ready markdown formatting", () => {
  it("pairs skills with trimmed EXTRA_INFO blocks using exact markdown spacing", () => {
    const skills = ["docker", "gitlab"];
    const extended: TextContent[] = [
      {
        type: "text",
        text: "prefix <extra_info>  Docker guide",
      },
      {
        type: "text",
        text: "\nwith two lines  </extra_info> ignored ",
      },
      {
        type: "text",
        text: "<EXTRA_INFO> GitLab guide </EXTRA_INFO> suffix",
      },
    ];

    expect(getSkillReadyContent(skills, extended)).toBe(
      [
        "",
        "",
        "**Triggered Skill Knowledge:**",
        "",
        "- **docker**",
        "",
        "Docker guide\nwith two lines",
        "",
        "- **gitlab**",
        "",
        "GitLab guide",
      ].join("\n"),
    );
  });

  it("renders a skill without extra spacing when no matching block exists", () => {
    const extended = makeTextContent(wrapExtraInfo("Docker guide"));

    expect(getSkillReadyContent(["docker", "gitlab"], extended)).toBe(
      [
        "",
        "",
        "**Triggered Skill Knowledge:**",
        "",
        "- **docker**",
        "",
        "Docker guide",
        "",
        "- **gitlab**",
      ].join("\n"),
    );
  });

  it("ignores unpaired EXTRA_INFO blocks when skills are present", () => {
    const extended = makeTextContent(
      `${wrapExtraInfo("Docker guide")}${wrapExtraInfo("Unused guide")}`,
    );

    expect(getSkillReadyContent(["docker"], extended)).toBe(
      "\n\n**Triggered Skill Knowledge:**\n\n- **docker**\n\nDocker guide",
    );
  });

  it("formats extracted blocks as numbered-free extended content without skills", () => {
    const extended = makeTextContent(
      `${wrapExtraInfo(" First block ")}${wrapExtraInfo("Second block")}`,
    );

    expect(getSkillReadyContent([], extended)).toBe(
      "\n\n**Extended Content:**\n\nFirst block\n\nSecond block",
    );
  });

  it("returns empty content when no complete non-empty EXTRA_INFO block exists", () => {
    const extended = makeTextContent(
      `plain text ${wrapExtraInfo("   ")} <EXTRA_INFO>unclosed`,
    );

    expect(getSkillReadyContent([], extended)).toBe("");
  });

  it("ignores non-text runtime entries while joining surrounding text", () => {
    const extended = [
      { type: "text", text: "<EXTRA_INFO>joined" },
      {
        type: "image",
        image_urls: ["data:image/png;base64,ignored"],
        text: "<EXTRA_INFO>poisoned image text</EXTRA_INFO>",
      },
      { type: "text", text: " block</EXTRA_INFO>" },
    ] as unknown as TextContent[];

    expect(getSkillReadyContent([], extended)).toBe(
      "\n\n**Extended Content:**\n\njoined block",
    );
  });

  it("falls back safely when legacy runtime data omits both arrays", () => {
    const missingSkills = undefined as unknown as string[];
    const missingContent = undefined as unknown as TextContent[];

    expect(getSkillReadyContent(missingSkills, missingContent)).toBe("");
    expect(getSkillReadyItems(missingSkills, missingContent)).toEqual([]);
  });
});
