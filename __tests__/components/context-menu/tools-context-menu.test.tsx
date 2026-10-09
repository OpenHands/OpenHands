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

describe("ToolsContextMenu - Switch agent profile", () => {
  it("hides the Switch agent profile item unless the caller enables it", () => {
    // The gate (pre-start only + profiles available) is owned by
    // ChatInputActions; without the prop the item — and its provider-backed
    // submenu content — must not mount.
    render(
      <ToolsContextMenu
        onClose={() => {}}
        onShowSkills={() => {}}
        onShowPlugins={() => {}}
        onShowHooks={() => {}}
        onShowAgentTools={() => {}}
      />,
    );

    expect(
      screen.queryByTestId("switch-agent-profile-button"),
    ).not.toBeInTheDocument();
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

describe("ToolsContextMenu - Submenu viewport placement (#18063)", () => {
  it("positions the macros submenu inside the viewport near screen edges", async () => {
    window.innerWidth = 390;
    window.innerHeight = 844;

    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(
      function (this: HTMLElement) {
        if (
          this.dataset?.testid === "macros-submenu" ||
          this.firstElementChild?.getAttribute("data-testid") ===
            "macros-submenu"
        ) {
          return DOMRect.fromRect({ x: 0, y: 0, width: 220, height: 200 });
        }
        return DOMRect.fromRect({ x: 16, y: 550, width: 194, height: 36 });
      },
    );

    render(
      <ToolsContextMenu
        onClose={() => {}}
        onShowSkills={() => {}}
        onShowPlugins={() => {}}
        onShowHooks={() => {}}
        onShowAgentTools={() => {}}
      />,
    );

    await userEvent.click(screen.getByTestId("macros-button"));

    const submenu = screen.getByTestId("macros-submenu");
    const submenuWrapper = submenu.parentElement;
    expect(submenuWrapper).toHaveClass("visible");
    expect(submenuWrapper).toHaveStyle({
      left: "146px",
      top: "-4px",
    });
  });
});
