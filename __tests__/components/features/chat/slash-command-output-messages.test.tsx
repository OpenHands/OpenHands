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
});
