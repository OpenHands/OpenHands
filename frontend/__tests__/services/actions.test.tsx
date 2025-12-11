import { describe, it, expect, vi, beforeEach } from "vitest";
import ActionType from "#/types/action-type";
import { ActionMessage } from "#/types/message";
import { useCommandStore } from "#/state/command-store";

const mockDispatch = vi.fn();
const mockAppendInput = vi.fn();

vi.mock("#/store", () => ({
  default: {
    dispatch: mockDispatch,
  },
}));

describe("handleActionMessage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useCommandStore.setState({
      appendInput: mockAppendInput,
    });
  });

  it("should handle RUN actions by adding input to terminal", async () => {
    const { handleActionMessage } = await import("#/services/actions");

    const runAction: ActionMessage = {
      id: 1,
      source: "agent",
      action: ActionType.RUN,
      args: {
        command: "ls -la",
      },
      message: "Running command: ls -la",
      timestamp: "2023-01-01T00:00:00Z",
    };

    // Handle the action
    handleActionMessage(runAction);

    // Check that appendInput was called with the command
    expect(mockAppendInput).toHaveBeenCalledWith("ls -la");
    expect(mockDispatch).not.toHaveBeenCalled();
  });

  it("should handle RUN_IPYTHON actions as no-op (Jupyter removed)", async () => {
    const { handleActionMessage } = await import("#/services/actions");

    const ipythonAction: ActionMessage = {
      id: 2,
      source: "agent",
      action: ActionType.RUN_IPYTHON,
      args: {
        code: "print('Hello from Jupyter!')",
      },
      message:
        "Running Python code interactively: print('Hello from Jupyter!')",
      timestamp: "2023-01-01T00:00:00Z",
    };

    // Handle the action
    handleActionMessage(ipythonAction);

    // Jupyter functionality has been removed, so nothing should be called
    expect(mockAppendInput).not.toHaveBeenCalled();
  });

  it("should not process hidden actions", async () => {
    const { handleActionMessage } = await import("#/services/actions");

    const hiddenAction: ActionMessage = {
      id: 3,
      source: "agent",
      action: ActionType.RUN,
      args: {
        command: "secret command",
        hidden: "true",
      },
      message: "Running command: secret command",
      timestamp: "2023-01-01T00:00:00Z",
    };

    // Handle the action
    handleActionMessage(hiddenAction);

    // Check that nothing was dispatched or called
    expect(mockDispatch).not.toHaveBeenCalled();
    expect(mockAppendInput).not.toHaveBeenCalled();
  });
});
