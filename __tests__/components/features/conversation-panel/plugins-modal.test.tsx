import { screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import {
  removeStoredConversationMetadata,
  setStoredConversationMetadata,
} from "#/api/conversation-metadata-store";
import { PluginsModal } from "#/components/features/conversation-panel/plugins-modal";

const CONVERSATION_ID = "conv-plugins-modal";

afterEach(() => removeStoredConversationMetadata(CONVERSATION_ID));

function renderModal() {
  return renderWithProviders(<PluginsModal onClose={vi.fn()} />, {
    navigation: { conversationId: CONVERSATION_ID },
  });
}

describe("PluginsModal", () => {
  it("lists the plugins attached to the conversation with their source and ref", () => {
    setStoredConversationMetadata(CONVERSATION_ID, {
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
      plugins: [
        {
          source: "github:OpenHands/extensions",
          ref: "main",
          repo_path: "plugins/city-weather",
        },
      ],
    });

    renderModal();

    expect(
      screen.getByTestId("active-plugin-city-weather"),
    ).toBeInTheDocument();
    expect(screen.getByText("OpenHands/extensions @ main")).toBeInTheDocument();
  });

  it("shows the plugin's name when the source is not a useful label", () => {
    // A locally-installed plugin: the source ("local") can't be derived into a
    // meaningful name, so the explicit name must be displayed.
    setStoredConversationMetadata(CONVERSATION_ID, {
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
      plugins: [
        { source: "local", ref: null, repo_path: null, name: "city-weather" },
      ],
    });

    renderModal();

    expect(
      screen.getByTestId("active-plugin-city-weather"),
    ).toBeInTheDocument();
  });

  it("renders a normalized 'Local' label for locally-installed plugins instead of the raw source", () => {
    // The agent-server records local installs two ways: the "local" sentinel
    // and an absolute filesystem path. Both should read the same, and the path
    // (which leaks the home dir) must not be shown verbatim.
    setStoredConversationMetadata(CONVERSATION_ID, {
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
      plugins: [
        { source: "local", ref: null, repo_path: null, name: "city-weather" },
        {
          source:
            "/Users/me/.openhands/cache/skills/public-skills/plugins/magic-test",
          ref: null,
          repo_path: null,
          name: "magic-test",
        },
      ],
    });

    renderModal();

    expect(screen.getAllByText("PLUGINS_MODAL$SOURCE_LOCAL")).toHaveLength(2);
    expect(
      screen.queryByText(
        "/Users/me/.openhands/cache/skills/public-skills/plugins/magic-test",
      ),
    ).not.toBeInTheDocument();
  });

  it("shows the empty state when no plugins are attached", () => {
    setStoredConversationMetadata(CONVERSATION_ID, {
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
    });

    renderModal();

    expect(screen.getByText("PLUGINS_MODAL$EMPTY")).toBeInTheDocument();
  });
});
