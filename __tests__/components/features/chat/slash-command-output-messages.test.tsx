import { beforeEach, describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import { SlashCommandOutputMessages } from "#/components/features/chat/slash-command-output-messages";
import { useSlashCommandOutputStore } from "#/stores/slash-command-output-store";
import { BUILT_IN_COMMANDS } from "#/utils/constants";

const CONVERSATION_ID = "conv-1";

describe("SlashCommandOutputMessages", () => {
  beforeEach(() => {
    useSlashCommandOutputStore.getState().clearAll();
  });

  it("renders help entries only at the requested anchor", () => {
    useSlashCommandOutputStore
      .getState()
      .showHelp(CONVERSATION_ID, "event-1", BUILT_IN_COMMANDS);

    renderWithProviders(
      <SlashCommandOutputMessages
        conversationId={CONVERSATION_ID}
        anchorEventId="event-1"
      />,
    );

    expect(
      screen.getByTestId("slash-command-output-messages"),
    ).toHaveTextContent("SLASH_COMMAND$HELP_TITLE");
    expect(screen.getByText("/help")).toBeInTheDocument();
  });

  it("renders nothing when no entries match the anchor", () => {
    useSlashCommandOutputStore
      .getState()
      .showHelp(CONVERSATION_ID, "event-1", BUILT_IN_COMMANDS);

    renderWithProviders(
      <SlashCommandOutputMessages
        conversationId={CONVERSATION_ID}
        anchorEventId="event-2"
      />,
    );

    expect(
      screen.queryByTestId("slash-command-output-messages"),
    ).not.toBeInTheDocument();
  });

  it("renders skills, hooks, and MCP servers in separate sections", () => {
    useSlashCommandOutputStore.getState().showSkills(CONVERSATION_ID, null, {
      skills: [
        {
          name: "code-search",
          type: "agentskills",
          source: "/workspace/.agents/skills/code-search/SKILL.md",
          content:
            "---\ndescription: Search this workspace without dumping the full skill body\n---\n# Code Search\n\nLong implementation instructions that should stay hidden.",
          triggers: ["/code-search"],
        },
      ],
      hooks: [
        {
          event_type: "pre_tool_use",
          matchers: [
            {
              matcher: "terminal",
              hooks: [{ type: "command", command: "npm test", timeout: 30 }],
            },
          ],
        },
      ],
      mcpServers: [
        {
          id: "stdio-0",
          type: "stdio",
          name: "filesystem",
          command: "npx",
        },
      ],
    });

    renderWithProviders(
      <SlashCommandOutputMessages
        conversationId={CONVERSATION_ID}
        anchorEventId={null}
      />,
    );

    expect(screen.getByText("code-search")).toBeInTheDocument();
    expect(
      screen.getByTestId("slash-skill-description-code-search"),
    ).toHaveTextContent(
      "Search this workspace without dumping the full skill body",
    );
    expect(
      screen.queryByText("Long implementation instructions that should stay hidden."),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("skill-type-badge-agentskills"),
    ).toBeInTheDocument();
    expect(screen.getByText("pre_tool_use")).toBeInTheDocument();
    expect(screen.getByText("filesystem")).toBeInTheDocument();
  });

  it("renders an explicit empty state when no extensions are loaded", () => {
    useSlashCommandOutputStore.getState().showSkills(CONVERSATION_ID, null, {
      skills: [],
      hooks: [],
      mcpServers: [],
    });

    renderWithProviders(
      <SlashCommandOutputMessages
        conversationId={CONVERSATION_ID}
        anchorEventId={null}
      />,
    );

    expect(screen.getByText("SLASH_COMMAND$NO_SKILLS")).toBeInTheDocument();
  });
});
