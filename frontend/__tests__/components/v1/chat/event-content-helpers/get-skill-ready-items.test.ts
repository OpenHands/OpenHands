import { describe, expect, it } from "vitest";
import { getSkillReadyItems } from "#/components/v1/chat/event-content-helpers/get-skill-ready-content";
import { TextContent } from "#/types/v1/core/base/common";

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
