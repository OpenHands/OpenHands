import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { compile } from "@tailwindcss/node";
import { parse } from "postcss";
import { afterEach, describe, it, expect, vi } from "vitest";
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
  it("keeps the Skills icon's unitless stroke in the rendered menu", async () => {
    render(
      <ToolsContextMenu
        onClose={() => {}}
        onShowSkills={() => {}}
        onShowPlugins={() => {}}
        onShowHooks={() => {}}
        onShowAgentTools={() => {}}
      />,
    );
    const icon = screen.getByTestId("show-skills-button").querySelector("svg");
    if (!icon) throw new Error("Expected the menu's Skills icon");
    const compiler = await compile(
      await readFile(resolve("src/tailwind.css"), "utf8"),
      { base: resolve("src"), onDependency: () => {} },
    );
    const rules: string[] = [];
    parse(compiler.build([...icon.classList])).walkDecls(
      "stroke-width",
      (declaration) => {
        const rule = declaration.parent;
        if (rule?.type === "rule" && icon.matches(rule.selector)) {
          rules.push(`${rule.selector}{${declaration.toString()}}`);
        }
      },
    );
    const style = document.createElement("style");
    style.textContent = rules.join("\n");
    document.head.appendChild(style);
    try {
      expect(getComputedStyle(icon).strokeWidth).toBe("1.75");
    } finally {
      style.remove();
    }
  });

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

describe("ToolsContextMenu - Macros submenu placement", () => {
  // jsdom has no layout: give the Macros row and its submenu the geometry the
  // browser renders (row at x=21, 240px wide; submenu 201px wide beside it).
  const ROW_LEFT = 21;
  const ROW_WIDTH = 240;
  const SUBMENU_WIDTH = 201;
  const VIEWPORT_GUTTER = 8;

  const renderAtViewportWidth = (width: number) => {
    vi.spyOn(window, "innerWidth", "get").mockReturnValue(width);
    const isFlyout = (el: HTMLElement) =>
      el.querySelector(":scope > [data-testid='macros-submenu']") !== null;
    const realRect = HTMLElement.prototype.getBoundingClientRect;
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(
      function getRect(this: HTMLElement) {
        return [...this.children].some(
          (child) => child instanceof HTMLElement && isFlyout(child),
        )
          ? DOMRect.fromRect({
              x: ROW_LEFT,
              y: 0,
              width: ROW_WIDTH,
              height: 36,
            })
          : realRect.call(this);
      },
    );
    vi.spyOn(HTMLElement.prototype, "offsetLeft", "get").mockImplementation(
      function getOffsetLeft(this: HTMLElement) {
        return isFlyout(this) ? ROW_WIDTH + 1 : 0;
      },
    );
    vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockImplementation(
      function getOffsetWidth(this: HTMLElement) {
        return this.dataset.testid === "macros-submenu" ? SUBMENU_WIDTH : 0;
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
  };

  // Where the submenu lands on screen: its beside-the-row position plus the
  // horizontal shift the flyout applies through its CSS variable.
  const renderedSubmenuLeft = () => {
    const flyout = screen.getByTestId("macros-submenu").parentElement!;
    const shift =
      parseFloat(flyout.style.getPropertyValue("--tools-submenu-shift")) || 0;
    return ROW_LEFT + ROW_WIDTH + 1 + shift;
  };

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("keeps the Macros submenu inside a phone-width viewport", async () => {
    renderAtViewportWidth(320);
    await userEvent.click(screen.getByTestId("macros-button"));

    const left = renderedSubmenuLeft();
    expect(left).toBeGreaterThanOrEqual(VIEWPORT_GUTTER);
    expect(left + SUBMENU_WIDTH).toBeLessThanOrEqual(320 - VIEWPORT_GUTTER);
  });

  it("leaves the Macros submenu beside the menu when the viewport has room", () => {
    renderAtViewportWidth(1440);

    expect(renderedSubmenuLeft()).toBe(ROW_LEFT + ROW_WIDTH + 1);
  });
});
