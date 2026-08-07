import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { resetCommandStore, useCommandStore } from "#/stores/command-store";
import { useAgentState } from "#/hooks/use-agent-state";
import { AgentState } from "#/types/agent-state";

vi.mock("#/hooks/use-agent-state");

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: {
      id: "test-conversation-id",
      conversation_url: "http://localhost:3000",
      session_api_key: "test-key",
      workspace: { working_dir: "/projects/odysseus" },
    },
    isFetched: true,
  }),
}));

vi.mock("#/hooks/use-bash-command-runner", () => ({
  useBashCommandRunner: () =>
    vi.fn(async () => ({ exit_code: 0, stdout: "", stderr: "" })),
}));

vi.mock("react-i18next", async (importOriginal) => {
  const real = await importOriginal<typeof import("react-i18next")>();
  return {
    ...real,
    useTranslation: () => ({
      t: (key: string, opts?: { number?: number }) =>
        opts?.number != null ? `${key}:${opts.number}` : key,
    }),
  };
});

const mockTerminalInstance = {
  open: vi.fn(),
  write: vi.fn(),
  writeln: vi.fn(),
  dispose: vi.fn(),
  loadAddon: vi.fn(),
  onData: vi.fn(() => ({ dispose: vi.fn() })),
  element: document.createElement("div"),
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

describe("Terminal empty state + tabs", () => {
  beforeEach(() => {
    resetCommandStore("test-conversation-id");
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.RUNNING,
    });
    global.ResizeObserver = vi.fn(function MockResizeObserver() {
      return {
        observe: vi.fn(),
        disconnect: vi.fn(),
      };
    }) as unknown as typeof ResizeObserver;
    vi.clearAllMocks();
  });

  it("shows an interactive prompt when runtime is active and there is no output", async () => {
    renderWithProviders(<Terminal />);

    expect(screen.queryByText("TERMINAL$NO_OUTPUT")).not.toBeInTheDocument();
    await waitFor(() => {
      expect(mockTerminalInstance.write).toHaveBeenCalledWith("$ ");
    });
  });

  it("keeps the terminal mounted when commands exist", () => {
    resetCommandStore("test-conversation-id", [
      { type: "output", content: "hello" },
    ]);

    renderWithProviders(<Terminal />);

    expect(screen.queryByText("TERMINAL$NO_OUTPUT")).not.toBeInTheDocument();
    expect(mockTerminalInstance.open).toHaveBeenCalled();
  });

  it("shows the runtime waiting state when inactive", () => {
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.LOADING,
    });

    renderWithProviders(<Terminal />);

    expect(screen.queryByText("TERMINAL$NO_OUTPUT")).not.toBeInTheDocument();
    expect(screen.getByTestId("runtime-waiting")).toBeInTheDocument();
  });

  it("can open a second terminal tab with an independent buffer", async () => {
    const user = userEvent.setup();
    resetCommandStore("test-conversation-id", [
      { type: "output", content: "first-tab" },
    ]);

    renderWithProviders(<Terminal />);

    expect(screen.getByTestId("terminal-tab-1")).toBeInTheDocument();
    await user.click(screen.getByTestId("terminal-add-tab"));

    expect(screen.getByTestId("terminal-tab-2")).toBeInTheDocument();
    expect(useCommandStore.getState().commands).toEqual([]);

    useCommandStore.getState().appendOutput("second-tab");
    expect(useCommandStore.getState().commands).toEqual([
      { type: "output", content: "second-tab", alreadyDisplayed: undefined },
    ]);
  });
});
