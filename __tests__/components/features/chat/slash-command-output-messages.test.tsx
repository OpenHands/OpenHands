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

  it("renders help and skills entries only at the requested anchor", () => {
    useSlashCommandOutputStore
      .getState()
      .showHelp(CONVERSATION_ID, "event-1", BUILT_IN_COMMANDS);
    useSlashCommandOutputStore
      .getState()
      .showSkills(CONVERSATION_ID, "event-2", {
        skills: [
          {
            name: "code-search",
            type: "agentskills",
            source: "project",
            description: "Search this workspace",
            triggers: ["/code-search"],
          },
        ],
        hooks: [],
        mcpServers: [],
      });

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
    expect(screen.queryByText("code-search")).not.toBeInTheDocument();
  });

  it("renders skills, hooks, and MCP servers in separate sections", () => {
    useSlashCommandOutputStore.getState().showSkills(CONVERSATION_ID, null, {
      skills: [
        {
          name: "code-search",
          type: "agentskills",
          source: "project",
          description: "Search this workspace",
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
