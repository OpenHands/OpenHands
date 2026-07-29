import { screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { useCommandStore } from "#/stores/command-store";
import { useAgentState } from "#/hooks/use-agent-state";
import { AgentState } from "#/types/agent-state";

vi.mock("#/hooks/use-agent-state");

const mockTerminalInstance = {
  open: vi.fn(),
  write: vi.fn(),
  writeln: vi.fn(),
  dispose: vi.fn(),
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
    useCommandStore.setState({ commands: [] });
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

  it("hides the empty state when terminal commands exist", () => {
    useCommandStore.setState({
      commands: [{ type: "output", content: "hello" }],
    });

    renderWithProviders(<Terminal />);

    expect(screen.queryByText("TERMINAL$NO_OUTPUT")).not.toBeInTheDocument();
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
