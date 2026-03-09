import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeAll } from "vitest";
import { renderWithProviders } from "test-utils";
import {
  SlashCommandMenu,
  getSkillDescription,
  stripMarkdown,
} from "#/components/features/chat/components/slash-command-menu";
import { SlashCommandItem } from "#/hooks/chat/use-slash-command";

// jsdom does not implement scrollIntoView
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn();
});

const makeItem = (
  name: string,
  command: string,
  content: string = "",
): SlashCommandItem => ({
  skill: {
    name,
    type: "agentskills" as const,
    content,
    triggers: [command],
  },
  command,
});

const defaultItems: SlashCommandItem[] = [
  makeItem("code-search", "/code-search", "Search code semantically."),
  makeItem("random-number", "/random-number", "Generate a random number."),
  makeItem(
    "init",
    "/init",
    "---\nname: init\ndescription: Initialize a project\n---\n## Usage\nRun /init to start.",
  ),
];

describe("SlashCommandMenu", () => {
  it("renders nothing when items is empty", () => {
    const { container } = renderWithProviders(
      <SlashCommandMenu items={[]} selectedIndex={0} onSelect={vi.fn()} />,
    );
    expect(container.innerHTML).toBe("");
  });

  it("renders all items with slash commands as primary text", () => {
    renderWithProviders(
      <SlashCommandMenu
        items={defaultItems}
        selectedIndex={0}
        onSelect={vi.fn()}
      />,
    );

    expect(screen.getByText("/code-search")).toBeInTheDocument();
    expect(screen.getByText("/random-number")).toBeInTheDocument();
    expect(screen.getByText("/init")).toBeInTheDocument();
  });

  it("marks the selected item with aria-selected", () => {
    renderWithProviders(
      <SlashCommandMenu
        items={defaultItems}
        selectedIndex={1}
        onSelect={vi.fn()}
      />,
    );

    const options = screen.getAllByRole("option");
    expect(options[0]).toHaveAttribute("aria-selected", "false");
    expect(options[1]).toHaveAttribute("aria-selected", "true");
    expect(options[2]).toHaveAttribute("aria-selected", "false");
  });

  it("calls onSelect on mouseDown", async () => {
    const onSelect = vi.fn();
    const user = userEvent.setup();

    renderWithProviders(
      <SlashCommandMenu
        items={defaultItems}
        selectedIndex={0}
        onSelect={onSelect}
      />,
    );

    const options = screen.getAllByRole("option");
    await user.click(options[1]);

    expect(onSelect).toHaveBeenCalledWith(defaultItems[1]);
  });

  it("displays skill descriptions", () => {
    renderWithProviders(
      <SlashCommandMenu
        items={defaultItems}
        selectedIndex={0}
        onSelect={vi.fn()}
      />,
    );

    // First item: first-sentence extraction
    expect(screen.getByText("Search code semantically.")).toBeInTheDocument();

    // Third item: frontmatter description extraction
    expect(screen.getByText("Initialize a project")).toBeInTheDocument();
  });

  it("has an accessible listbox role and translated aria-label", () => {
    renderWithProviders(
      <SlashCommandMenu
        items={defaultItems}
        selectedIndex={0}
        onSelect={vi.fn()}
      />,
    );

    const listbox = screen.getByRole("listbox");
    expect(listbox).toBeInTheDocument();
    // In test env, translation key is returned as-is
    expect(listbox).toHaveAttribute("aria-label", "CHAT_INTERFACE$COMMANDS");
  });
});

describe("getSkillDescription", () => {
  it("extracts description from YAML frontmatter", () => {
    const content =
      "---\nname: test\ndescription: A test skill\n---\n## Usage\nDetails here.";
    expect(getSkillDescription(content)).toBe("A test skill");
  });

  it("strips double quotes from frontmatter description", () => {
    const content = '---\ndescription: "Quoted description"\n---\nBody.';
    expect(getSkillDescription(content)).toBe("Quoted description");
  });

  it("strips single quotes from frontmatter description", () => {
    const content = "---\ndescription: 'Single quoted'\n---\nBody.";
    expect(getSkillDescription(content)).toBe("Single quoted");
  });

  it("falls back to first meaningful line when no frontmatter", () => {
    const content = "# Title\n\nThis is a description.";
    expect(getSkillDescription(content)).toBe("This is a description.");
  });

  it("falls back to first sentence from body when frontmatter has no description", () => {
    const content =
      "---\nname: test\ntriggers: ['/test']\n---\nA helpful tool. It does things.";
    expect(getSkillDescription(content)).toBe("A helpful tool.");
  });

  it("skips headers and empty lines", () => {
    const content = "\n\n# Header\n## Subheader\n\nActual content here";
    expect(getSkillDescription(content)).toBe("Actual content here");
  });

  it("returns null for empty content", () => {
    expect(getSkillDescription("")).toBeNull();
  });

  it("returns null for content with only headers", () => {
    expect(getSkillDescription("# Title\n## Subtitle")).toBeNull();
  });

  it("returns the whole line when there is no sentence-ending punctuation", () => {
    const content = "A description without punctuation";
    expect(getSkillDescription(content)).toBe(
      "A description without punctuation",
    );
  });

  it("strips markdown from frontmatter description", () => {
    const content =
      '---\ndescription: "A **bold** and *italic* description"\n---\nBody.';
    expect(getSkillDescription(content)).toBe(
      "A bold and italic description",
    );
  });

  it("strips markdown from body fallback", () => {
    const content = "# Title\n\nUse `code` and [links](http://example.com).";
    expect(getSkillDescription(content)).toBe("Use code and links.");
  });
});

describe("stripMarkdown", () => {
  it("strips bold syntax", () => {
    expect(stripMarkdown("a **bold** word")).toBe("a bold word");
  });

  it("strips italic syntax", () => {
    expect(stripMarkdown("an *italic* word")).toBe("an italic word");
  });

  it("strips bold-italic syntax", () => {
    expect(stripMarkdown("***both***")).toBe("both");
  });

  it("strips inline code", () => {
    expect(stripMarkdown("run `npm test` now")).toBe("run npm test now");
  });

  it("strips links", () => {
    expect(stripMarkdown("see [docs](http://example.com)")).toBe("see docs");
  });

  it("strips images", () => {
    expect(stripMarkdown("![alt text](image.png)")).toBe("alt text");
  });

  it("strips strikethrough", () => {
    expect(stripMarkdown("~~removed~~")).toBe("removed");
  });

  it("strips underscore emphasis", () => {
    expect(stripMarkdown("__bold__ and _italic_")).toBe("bold and italic");
  });

  it("returns plain text unchanged", () => {
    expect(stripMarkdown("plain text")).toBe("plain text");
  });
});
