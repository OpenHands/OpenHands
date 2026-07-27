import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { adaptSystemMessage } from "#/utils/system-message-adapter";
import { EventState } from "#/stores/use-event-store";
import { SystemMessageModal } from "#/components/features/conversation-panel/system-message-modal";
import { ToolsContextMenu } from "#/components/features/controls/tools-context-menu";

const v1Event: EventState["events"] = [
  {
    id: "v1-id",
    timestamp: "2025-12-30T12:00:00Z",
    source: "agent",
    system_prompt: {
      type: "text",
      text: "v1 prompt",
    },
    tools: [
      {
        type: "function",
        function: {
          name: "bash",
          description: "Execute bash",
          parameters: {},
        },
      },
    ],
  },
];

const adaptedResult = adaptSystemMessage(v1Event);

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({ data: { conversation_version: "V1" } }),
}));

vi.mock("#/hooks/use-user-providers", () => ({
  useUserProviders: () => ({ providers: ["test"] }),
}));

describe("SystemMessage UI Rendering", () => {
  it("should render the 'Show Agent Tools' button in the context menu", () => {
    render(
      <ToolsContextMenu
        onClose={() => {}}
        onShowSkills={() => {}}
        onShowPlugins={() => {}}
        onShowHooks={() => {}}
        onShowAgentTools={() => {}}
      />,
    );

    expect(screen.getByTestId("show-agent-tools-button")).toBeInTheDocument();
  });

  it("should display the adapted v1 system prompt content correctly", () => {
    render(
      <SystemMessageModal
        isOpen
        onClose={() => {}}
        systemMessage={adaptedResult}
      />,
    );

    const messageElement = screen.getByText("v1 prompt");

    expect(messageElement).toBeDefined();
    expect(messageElement).toBeVisible();
  });
});

describe("ToolsContextMenu - Show Plugins", () => {
  it("renders the Show Plugins item and calls onShowPlugins when clicked", async () => {
    const onShowPlugins = vi.fn();
    render(
      <ToolsContextMenu
        onClose={() => {}}
        onShowSkills={() => {}}
        onShowPlugins={onShowPlugins}
        onShowHooks={() => {}}
        onShowAgentTools={() => {}}
        shouldShowPlugins
      />,
    );

    await userEvent.click(screen.getByTestId("show-plugins-button"));

    expect(onShowPlugins).toHaveBeenCalledTimes(1);
  });

  it("hides the Show Plugins item when the conversation has no attached plugins", () => {
    render(
      <ToolsContextMenu
        onClose={() => {}}
        onShowSkills={() => {}}
        onShowPlugins={() => {}}
        onShowHooks={() => {}}
        onShowAgentTools={() => {}}
      />,
    );

    expect(screen.queryByTestId("show-plugins-button")).not.toBeInTheDocument();
  });
});
