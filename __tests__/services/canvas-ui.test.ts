import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  CANVAS_UI_CLIENT_ACTION_KIND,
  CANVAS_UI_CLIENT_TOOL_NAME,
  LEGACY_CANVAS_UI_TOOL_NAME,
} from "#/constants/canvas-ui";
import { handleCanvasUIAction } from "#/services/canvas-ui";
import { useConversationStore } from "#/stores/conversation-store";
import { useFilesTabStore } from "#/stores/files-tab-store";
import type { CanvasUIAction } from "#/types/agent-server/core";
import { isCanvasUIActionEvent } from "#/types/agent-server/type-guards";

// Helper: build a CanvasUIAction without repeating the literal `kind`
// discriminator in every test case.
function action(overrides: Partial<CanvasUIAction>): CanvasUIAction {
  return { kind: "CanvasUIAction", ...overrides } as CanvasUIAction;
}

describe("handleCanvasUIAction", () => {
  beforeEach(() => {
    // Arrange (shared): collapsed right panel, no selected file. Lets us
    // observe both the tab/panel toggling and the path mutation that the
    // dispatcher performs.
    useConversationStore.setState({
      selectedTab: null,
      isRightPanelShown: false,
      hasRightPanelToggled: false,
    });
    useFilesTabStore.setState({
      selectedPath: null,
      selectedConversationId: null,
    });
  });

  it("navigate_to_file selects the files tab, reveals the panel, and sets selectedPath", () => {
    handleCanvasUIAction(
      action({ command: "navigate_to_file", path: "docs/intro.html" }),
      "conv-1",
    );

    const conv = useConversationStore.getState();
    expect(conv.selectedTab).toBe("files");
    expect(conv.isRightPanelShown).toBe(true);
    expect(useFilesTabStore.getState().selectedPath).toBe("docs/intro.html");
    expect(useFilesTabStore.getState().selectedConversationId).toBe("conv-1");
  });

  it("show_preview selects the files tab and the requested path", () => {
    handleCanvasUIAction(
      action({ command: "show_preview", path: "report.html" }),
    );

    expect(useConversationStore.getState().selectedTab).toBe("files");
    expect(useFilesTabStore.getState().selectedPath).toBe("report.html");
  });

  it("open_tab switches to a valid tab without touching selectedPath", () => {
    handleCanvasUIAction(action({ command: "open_tab", tab: "terminal" }));

    expect(useConversationStore.getState().selectedTab).toBe("terminal");
    expect(useFilesTabStore.getState().selectedPath).toBeNull();
  });

  it("open_tab ignores unknown tab values and surfaces a warning for debuggability", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    try {
      handleCanvasUIAction(action({ command: "open_tab", tab: "not_a_tab" }));

      expect(useConversationStore.getState().selectedTab).toBeNull();
      expect(useConversationStore.getState().isRightPanelShown).toBe(false);
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining("not_a_tab"),
      );
    } finally {
      warnSpy.mockRestore();
    }
  });

  it("open_tab routes the removed vscode tab to files", () => {
    handleCanvasUIAction(action({ command: "open_tab", tab: "vscode" }));

    expect(useConversationStore.getState().selectedTab).toBe("files");
    expect(useConversationStore.getState().isRightPanelShown).toBe(true);
    expect(useFilesTabStore.getState().selectedPath).toBeNull();
  });

  it("navigate_to_file leaves selectedPath alone when no path is supplied", () => {
    useFilesTabStore.setState({ selectedPath: "previous.txt" });

    handleCanvasUIAction(action({ command: "navigate_to_file", path: null }));

    expect(useFilesTabStore.getState().selectedPath).toBe("previous.txt");
    expect(useConversationStore.getState().selectedTab).toBe("files");
  });
});

describe("isCanvasUIActionEvent", () => {
  function makeActionEvent(overrides: Record<string, unknown> = {}) {
    return {
      id: "evt-1",
      timestamp: "2026-05-13T00:00:00Z",
      source: "agent",
      action: { kind: "CanvasUIAction" },
      tool_name: LEGACY_CANVAS_UI_TOOL_NAME,
      tool_call_id: "call-1",
      ...overrides,
    };
  }

  it.each([
    ["CanvasUIAction", LEGACY_CANVAS_UI_TOOL_NAME],
    [CANVAS_UI_CLIENT_ACTION_KIND, CANVAS_UI_CLIENT_TOOL_NAME],
  ])("returns true for a %s ActionEvent from %s", (kind, toolName) => {
    expect(
      isCanvasUIActionEvent(
        makeActionEvent({
          action: { kind, command: "open_tab" },
          tool_name: toolName,
        }) as never,
      ),
    ).toBe(true);
  });

  it("returns false when tool_name belongs to a different tool", () => {
    expect(
      isCanvasUIActionEvent(
        makeActionEvent({ tool_name: "execute_bash" }) as never,
      ),
    ).toBe(false);
  });

  it("returns false for a non-action event (no action field)", () => {
    const observationEvent = {
      id: "evt-2",
      timestamp: "2026-05-13T00:00:00Z",
      source: "environment",
      observation: { kind: "ExecuteBashObservation" },
      tool_name: LEGACY_CANVAS_UI_TOOL_NAME,
      tool_call_id: "call-1",
    };

    expect(isCanvasUIActionEvent(observationEvent as never)).toBe(false);
  });
});
