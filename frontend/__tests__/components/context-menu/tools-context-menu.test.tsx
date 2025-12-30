import { describe, it, expect, vi } from "vitest";
import { adaptSystemMessage } from "#/utils/system-message-adapter";
import { EventState } from "#/stores/use-event-store";

//기존의 로직이 v0 를 기준으로 짜여져 있고 v1 타입에 대한 어댑팅을 추가한 것이기 때문에 v1 타입에 대한 테스트를 추가
//이 로직이 정상적으로 작동되면 system message 모달에서 v1 타입의 system prompt가 정상적으로 표시됨

import { render, screen } from "@testing-library/react";
import { SystemMessageModal } from "#/components/features/conversation-panel/system-message-modal";
import { ToolsContextMenu } from "#/components/features/controls/tools-context-menu";
import userEvent from "@testing-library/user-event";
import React from "react";

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
        onShowAgentTools={() => {}}
      ></ToolsContextMenu>,
    );

    expect(screen.getByTestId("show-agent-tools-button")).toBeInTheDocument();
  });

  it("should display the adapted v1 system prompt content correctly", () => {
    render(
      <SystemMessageModal
        isOpen={true}
        onClose={() => {}}
        systemMessage={adaptedResult}
      />,
    );

    const messageElement = screen.getByText("v1 prompt");

    expect(messageElement).toBeDefined();
    expect(messageElement).toBeVisible();
  });

  it("should open the system message modal when the agent tools button is clicked", async () => {
    const user = userEvent.setup();

    const TestWrapper = () => {
      const [isOpen, setIsOpen] = React.useState(false);
      return (
        <>
          <ToolsContextMenu
            onClose={() => setIsOpen(false)}
            onShowSkills={() => {}}
            onShowAgentTools={() => setIsOpen(true)}
          />
          <SystemMessageModal
            isOpen={isOpen}
            onClose={() => setIsOpen(false)}
            systemMessage={adaptedResult}
          />
        </>
      );
    };

    render(<TestWrapper />);

    expect(screen.queryByText("v1 prompt")).not.toBeInTheDocument();

    const button = screen.getByTestId("show-agent-tools-button");
    await user.click(button);

    expect(screen.queryByText("v1 prompt")).toBeInTheDocument();
    const modalContent = screen.getByText("v1 prompt");
    expect(modalContent).toBeInTheDocument();
    expect(modalContent).toBeVisible();
  });
});

describe("adaptSystemMessage", () => {
  it("should correctly adapt the v1 system_prompt event structure", () => {
    expect(adaptedResult).not.toBeNull();
    expect(adaptedResult?.content).toBe("v1 prompt");
  });

  it("should return null when no system message is present in events", () => {
    expect(adaptSystemMessage([])).toBeNull();
  });
});
