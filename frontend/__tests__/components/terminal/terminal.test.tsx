import { act, screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import { vi, describe, afterEach, it, expect } from "vitest";
import { Command, useCommandStore } from "#/state/command-store";
import Terminal from "#/components/features/terminal/terminal";

const renderTerminal = (commands: Command[] = []) => {
  // Set initial commands in Zustand store
  useCommandStore.setState({ commands });
  return renderWithProviders(<Terminal />);
};

describe.skip("Terminal", () => {
  // Terminal is now read-only - no user input functionality
  global.ResizeObserver = vi.fn().mockImplementation(() => ({
    observe: vi.fn(),
    disconnect: vi.fn(),
  }));

  const mockTerminal = {
    open: vi.fn(),
    write: vi.fn(),
    writeln: vi.fn(),
    dispose: vi.fn(),
    loadAddon: vi.fn(),
  };

  vi.mock("@xterm/xterm", async (importOriginal) => ({
    ...(await importOriginal<typeof import("@xterm/xterm")>()),
    Terminal: vi.fn().mockImplementation(() => mockTerminal),
  }));

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render a terminal", () => {
    renderTerminal();

    expect(screen.getByText("Terminal")).toBeInTheDocument();
    expect(mockTerminal.open).toHaveBeenCalledTimes(1);

    expect(mockTerminal.write).toHaveBeenCalledWith("$ ");
  });

  it("should load commands to the terminal", () => {
    renderTerminal([
      { type: "input", content: "INPUT" },
      { type: "output", content: "OUTPUT" },
    ]);

    expect(mockTerminal.writeln).toHaveBeenNthCalledWith(1, "INPUT");
    expect(mockTerminal.writeln).toHaveBeenNthCalledWith(2, "OUTPUT");
  });

  it("should write commands to the terminal", () => {
    renderTerminal();

    act(() => {
      useCommandStore.getState().appendInput("echo Hello");
      useCommandStore.getState().appendOutput("Hello");
    });

    expect(mockTerminal.writeln).toHaveBeenNthCalledWith(1, "echo Hello");
    expect(mockTerminal.writeln).toHaveBeenNthCalledWith(2, "Hello");

    act(() => {
      useCommandStore.getState().appendInput("echo World");
    });

    expect(mockTerminal.writeln).toHaveBeenNthCalledWith(3, "echo World");
  });

  it("should load and write commands to the terminal", () => {
    renderTerminal([
      { type: "input", content: "echo Hello" },
      { type: "output", content: "Hello" },
    ]);

    expect(mockTerminal.writeln).toHaveBeenNthCalledWith(1, "echo Hello");
    expect(mockTerminal.writeln).toHaveBeenNthCalledWith(2, "Hello");

    act(() => {
      useCommandStore.getState().appendInput("echo Hello");
    });

    expect(mockTerminal.writeln).toHaveBeenNthCalledWith(3, "echo Hello");
  });

  it("should end the line with a dollar sign after writing a command", () => {
    renderTerminal();

    act(() => {
      useCommandStore.getState().appendInput("echo Hello");
    });

    expect(mockTerminal.writeln).toHaveBeenCalledWith("echo Hello");
    expect(mockTerminal.write).toHaveBeenCalledWith("$ ");
  });

  it("should display a custom symbol if output contains a custom symbol", () => {
    renderTerminal([
      { type: "input", content: "echo Hello" },
      {
        type: "output",
        content:
          "Hello\r\n\r\n[Python Interpreter: /openhands/poetry/openhands-5O4_aCHf-py3.12/bin/python]\nopenhands@659478cb008c:/workspace $ ",
      },
    ]);

    expect(mockTerminal.writeln).toHaveBeenNthCalledWith(1, "echo Hello");
    expect(mockTerminal.writeln).toHaveBeenNthCalledWith(2, "Hello");
    expect(mockTerminal.write).toHaveBeenCalledWith(
      "\nopenhands@659478cb008c:/workspace $ ",
    );
  });

  // This test fails because it expects `disposeMock` to have been called before the component is unmounted.
  it.skip("should dispose the terminal on unmount", () => {
    const { unmount } = renderWithProviders(<Terminal />);

    expect(mockTerminal.dispose).not.toHaveBeenCalled();

    unmount();

    expect(mockTerminal.dispose).toHaveBeenCalledTimes(1);
  });
});
