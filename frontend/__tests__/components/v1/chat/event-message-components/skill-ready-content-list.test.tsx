import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect } from "vitest";
import { renderWithProviders } from "test-utils";
import { SkillReadyContentList } from "#/components/v1/chat/event-message-components/skill-ready-content-list";
import { SkillReadyItem } from "#/components/v1/chat/event-content-helpers/create-skill-ready-event";

const makeItems = (
  ...entries: [string, string][]
): SkillReadyItem[] =>
  entries.map(([name, content]) => ({ name, content }));

describe("SkillReadyContentList", () => {
  it("renders all skill names", () => {
    const items = makeItems(["docker", "content1"], ["gitlab", "content2"]);

    renderWithProviders(<SkillReadyContentList items={items} />);

    expect(screen.getByText("docker")).toBeInTheDocument();
    expect(screen.getByText("gitlab")).toBeInTheDocument();
  });

  it("renders the header label", () => {
    const items = makeItems(["docker", "content"]);

    renderWithProviders(<SkillReadyContentList items={items} />);

    expect(
      screen.getByText("SKILLS$TRIGGERED_SKILL_KNOWLEDGE"),
    ).toBeInTheDocument();
  });

  it("does not show content before clicking", () => {
    const items = makeItems(["docker", "Docker usage guide"]);

    renderWithProviders(<SkillReadyContentList items={items} />);

    expect(screen.queryByText("Docker usage guide")).not.toBeInTheDocument();
  });

  it("expands skill content on click", async () => {
    const user = userEvent.setup();
    const items = makeItems(["docker", "Docker usage guide"]);

    renderWithProviders(<SkillReadyContentList items={items} />);

    await user.click(screen.getByText("docker"));

    expect(screen.getByText("Docker usage guide")).toBeInTheDocument();
  });

  it("collapses skill content on second click", async () => {
    const user = userEvent.setup();
    const items = makeItems(["docker", "Docker usage guide"]);

    renderWithProviders(<SkillReadyContentList items={items} />);

    await user.click(screen.getByText("docker"));
    expect(screen.getByText("Docker usage guide")).toBeInTheDocument();

    await user.click(screen.getByText("docker"));
    expect(screen.queryByText("Docker usage guide")).not.toBeInTheDocument();
  });

  it("expands skills independently", async () => {
    const user = userEvent.setup();
    const items = makeItems(
      ["docker", "Docker guide"],
      ["gitlab", "GitLab guide"],
    );

    renderWithProviders(<SkillReadyContentList items={items} />);

    await user.click(screen.getByText("docker"));

    expect(screen.getByText("Docker guide")).toBeInTheDocument();
    expect(screen.queryByText("GitLab guide")).not.toBeInTheDocument();
  });

  it("renders <important> content as bold text", async () => {
    const user = userEvent.setup();
    const content = "Some text <important>critical info</important> more text";
    const items = makeItems(["docker", content]);

    renderWithProviders(<SkillReadyContentList items={items} />);

    await user.click(screen.getByText("docker"));

    // The important text should be rendered as bold (strong element)
    const boldElement = screen.getByText("critical info");
    expect(boldElement).toBeInTheDocument();
    expect(boldElement.tagName).toBe("STRONG");
  });

  it("parses and displays file path from metadata", async () => {
    const user = userEvent.setup();
    const content = [
      "The following information has been included based on a keyword match for \"docker\".",
      "It may or may not be relevant to the user's request.",
      "Skill location: /home/openhands/.openhands/skills/docker/SKILL.md",
      "(Use this path to resolve relative file references)",
      "",
      "Docker Usage Guide",
    ].join("\n");
    const items = makeItems(["docker", content]);

    renderWithProviders(<SkillReadyContentList items={items} />);

    await user.click(screen.getByText("docker"));

    // File path rendered in code element
    expect(
      screen.getByText(
        "/home/openhands/.openhands/skills/docker/SKILL.md",
      ),
    ).toBeInTheDocument();
    // Actual skill body rendered
    expect(screen.getByText("Docker Usage Guide")).toBeInTheDocument();
    // Metadata preamble lines are not rendered as-is in the body
    expect(
      screen.queryByText("It may or may not be relevant"),
    ).not.toBeInTheDocument();
  });
});
