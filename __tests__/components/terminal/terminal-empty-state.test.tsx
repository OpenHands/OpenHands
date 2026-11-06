import { screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { useEventStore } from "#/stores/use-event-store";
import { createOtherActionEvent } from "test-utils";
import { useAgentState } from "#/hooks/use-agent-state";
import { AgentState } from "#/types/agent-state";

vi.mock("#/hooks/use-agent-state");

const mockTerminalInstance = {
  open: vi.fn(),
  write: vi.fn(),
  writeln: vi.fn(),
  dispose: vi.fn(),
  reset: vi.fn(),
  loadAddon: vi.fn(),
};

vi.mock("@xterm/xterm", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@xterm/xterm")>()),
  Terminal: vi.fn(function MockTerminal() {
    return mockTerminalInstance;
  }),
}));

vi.mock("@xterm/addon-fit", () => ({
  FitAddon: vi.fn(function MockFitAddon() {
    return { fit: vi.fn() };
  }),
}));

import { renderWithProviders } from "test-utils";
import Terminal from "#/components/features/terminal/terminal";

describe("Terminal empty state", () => {
  beforeEach(() => {
    useEventStore.getState().clearEvents();
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.RUNNING,
    });
    global.ResizeObserver = vi.fn(function MockResizeObserver() {
      return {
        observe: vi.fn(),
        disconnect: vi.fn(),
      };
    }) as unknown as typeof ResizeObserver;
  });

  it("shows the empty state when runtime is active and there is no output", () => {
    renderWithProviders(<Terminal />);

    expect(screen.getByText("TERMINAL$NO_OUTPUT")).toBeInTheDocument();
  });

  it("shows saved commands when the terminal is first opened after history loads", () => {
    useEventStore
      .getState()
      .addEvents([createOtherActionEvent("saved-command")]);

    renderWithProviders(<Terminal />);

    expect(screen.queryByText("TERMINAL$NO_OUTPUT")).not.toBeInTheDocument();
    expect(mockTerminalInstance.writeln).toHaveBeenCalledWith("echo test");
  });

  it("shows the runtime waiting state instead of the empty state when inactive", () => {
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.LOADING,
    });

    renderWithProviders(<Terminal />);

    expect(screen.queryByText("TERMINAL$NO_OUTPUT")).not.toBeInTheDocument();
    expect(screen.getByTestId("runtime-waiting")).toBeInTheDocument();
  });
});
