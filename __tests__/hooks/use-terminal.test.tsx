import { act } from "@testing-library/react";
import { beforeAll, describe, expect, it, vi, afterEach } from "vitest";
import { useTerminal } from "#/hooks/use-terminal";
import { Command, useCommandStore } from "#/stores/command-store";
import { renderWithProviders } from "../../test-utils";

// Mock useConversationWebSocket
vi.mock("#/contexts/conversation-websocket-context", () => ({
  useConversationWebSocket: () => null,
}));

function TestTerminalComponent() {
  const ref = useTerminal();
  return <div ref={ref} />;
}

// Canvas currently mounts one terminal tab at a time; this pair exercises the
// hook's defensive instance-isolation invariant for independent consumers.
function TestTerminalPair({ showFirst = true }: { showFirst?: boolean } = {}) {
  return (
    <>
      {showFirst && <TestTerminalComponent key="first" />}
      <TestTerminalComponent key="second" />
    </>
  );
}

// Terminal is read-only - no longer tests user input functionality
const mockTerminal = vi.hoisted(() => ({
  loadAddon: vi.fn(),
  open: vi.fn(),
  write: vi.fn(),
  writeln: vi.fn(),
  dispose: vi.fn(),
  reset: vi.fn(),
  element: document.createElement("div"),
}));

const mockFitAddon = vi.hoisted(() => ({
  fit: vi.fn(),
}));

// mock Terminal - use class for Vitest 4 constructor support
vi.mock("@xterm/xterm", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@xterm/xterm")>()),
  Terminal: class {
    loadAddon = mockTerminal.loadAddon;

    open = mockTerminal.open;

    write = mockTerminal.write;

    writeln = mockTerminal.writeln;

    dispose = mockTerminal.dispose;

    reset = mockTerminal.reset;

    element = mockTerminal.element;
  },
}));

// mock FitAddon
vi.mock("@xterm/addon-fit", () => ({
  FitAddon: class {
    fit = mockFitAddon.fit;
  },
}));

describe("useTerminal", () => {
  beforeAll(() => {
    // mock ResizeObserver - use class for Vitest 4 constructor support
    window.ResizeObserver = class {
      observe = vi.fn();

      unobserve = vi.fn();

      disconnect = vi.fn();
    } as unknown as typeof ResizeObserver;
  });

  afterEach(() => {
    vi.clearAllMocks();
    // Reset command store between tests
    useCommandStore.setState({ commands: [] });
  });

  it("should render", () => {
    renderWithProviders(<TestTerminalComponent />);
  });

  it("should render the commands in the terminal", () => {
    const commands: Command[] = [
      { content: "echo hello", type: "input" },
      { content: "hello", type: "output" },
    ];

    // Set commands in store before rendering to ensure they're picked up during initialization
    useCommandStore.setState({ commands });

    renderWithProviders(<TestTerminalComponent />);

    expect(mockTerminal.writeln).toHaveBeenNthCalledWith(1, "echo hello");
    expect(mockTerminal.writeln).toHaveBeenNthCalledWith(2, "hello");
  });

  it("should render new commands independently in multiple hook instances", () => {
    renderWithProviders(<TestTerminalPair />);

    act(() => {
      useCommandStore.setState({
        commands: [{ content: "echo hello", type: "input" }],
      });
    });

    expect(mockTerminal.writeln).toHaveBeenCalledTimes(2);
    expect(mockTerminal.writeln).toHaveBeenNthCalledWith(1, "echo hello");
    expect(mockTerminal.writeln).toHaveBeenNthCalledWith(2, "echo hello");
  });

  it("should initialize every terminal from preloaded commands", () => {
    const commands: Command[] = [
      { content: "echo hello", type: "input" },
      { content: "hello", type: "output" },
    ];
    useCommandStore.setState({ commands });

    renderWithProviders(<TestTerminalPair />);

    expect(mockTerminal.writeln).toHaveBeenCalledTimes(4);
  });

  it("should not reset a sibling terminal's cursor when one unmounts", () => {
    const commands: Command[] = [
      { content: "echo hello", type: "input" },
      { content: "hello", type: "output" },
    ];
    useCommandStore.setState({ commands });

    const { rerender } = renderWithProviders(<TestTerminalPair />);
    expect(mockTerminal.writeln).toHaveBeenCalledTimes(4);

    mockTerminal.writeln.mockClear();
    rerender(<TestTerminalPair showFirst={false} />);

    act(() => {
      useCommandStore.setState({
        commands: [...commands, { content: "done", type: "output" }],
      });
    });

    expect(mockTerminal.writeln).toHaveBeenCalledTimes(1);
    expect(mockTerminal.writeln).toHaveBeenCalledWith("done");
  });

  it("should replay command history when a terminal remounts", () => {
    const commands: Command[] = [
      { content: "echo hello", type: "input" },
      { content: "hello", type: "output" },
    ];
    useCommandStore.setState({ commands });

    const firstRender = renderWithProviders(<TestTerminalComponent />);
    expect(mockTerminal.writeln).toHaveBeenCalledTimes(2);

    firstRender.unmount();
    mockTerminal.writeln.mockClear();
    renderWithProviders(<TestTerminalComponent />);

    expect(mockTerminal.writeln).toHaveBeenCalledTimes(2);
  });

  it("should reset rendered history when the command store is cleared", () => {
    const originalCommands: Command[] = [
      { content: "echo hello", type: "input" },
      { content: "hello", type: "output" },
    ];
    useCommandStore.setState({ commands: originalCommands });

    renderWithProviders(<TestTerminalComponent />);
    expect(mockTerminal.writeln).toHaveBeenCalledTimes(2);

    mockTerminal.write.mockClear();
    mockTerminal.writeln.mockClear();
    act(() => {
      useCommandStore.getState().clearTerminal();
    });
    expect(mockTerminal.write).toHaveBeenCalledWith("\x1bc\x1b[?25l");

    act(() => {
      useCommandStore.getState().appendInput("echo fresh");
      useCommandStore.getState().appendOutput("fresh");
    });

    expect(mockTerminal.writeln).toHaveBeenCalledTimes(2);
    expect(mockTerminal.writeln).toHaveBeenNthCalledWith(1, "echo fresh");
    expect(mockTerminal.writeln).toHaveBeenNthCalledWith(2, "fresh");
    expect(mockTerminal.write.mock.invocationCallOrder[0]).toBeLessThan(
      mockTerminal.writeln.mock.invocationCallOrder[0],
    );
  });

  it("should reset rendered history when the command store is cleared and reseeded in one update", () => {
    const originalCommands: Command[] = [
      { content: "echo hello", type: "input" },
      { content: "hello", type: "output" },
    ];
    useCommandStore.setState({ commands: originalCommands });

    renderWithProviders(<TestTerminalComponent />);
    expect(mockTerminal.writeln).toHaveBeenCalledTimes(2);

    mockTerminal.write.mockClear();
    mockTerminal.writeln.mockClear();
    act(() => {
      useCommandStore.getState().clearTerminal();
      useCommandStore.getState().appendInput("echo fresh");
      useCommandStore.getState().appendOutput("fresh");
    });

    expect(mockTerminal.write).toHaveBeenCalledWith("\x1bc\x1b[?25l");
    expect(mockTerminal.writeln).toHaveBeenCalledTimes(2);
    expect(mockTerminal.writeln).toHaveBeenNthCalledWith(1, "echo fresh");
    expect(mockTerminal.writeln).toHaveBeenNthCalledWith(2, "fresh");
    expect(mockTerminal.write.mock.invocationCallOrder[0]).toBeLessThan(
      mockTerminal.writeln.mock.invocationCallOrder[0],
    );
  });

  it("should clear writes still queued when history is replaced", () => {
    const queuedWrites: (() => void)[] = [];
    let renderedLines: string[] = [];
    mockTerminal.write.mockImplementation((data: string) => {
      queuedWrites.push(() => {
        if (data.includes("\x1bc")) renderedLines = [];
      });
    });
    mockTerminal.writeln.mockImplementation((data: string) => {
      queuedWrites.push(() => renderedLines.push(data));
    });
    mockTerminal.reset.mockImplementation(() => {
      renderedLines = [];
    });

    try {
      useCommandStore.setState({
        commands: [{ content: "old conversation", type: "output" }],
      });
      renderWithProviders(<TestTerminalComponent />);

      act(() => {
        useCommandStore.getState().clearTerminal();
        useCommandStore.getState().appendOutput("current conversation");
      });

      queuedWrites.forEach((write) => write());
      expect(renderedLines).toEqual(["current conversation"]);
    } finally {
      mockTerminal.write.mockReset();
      mockTerminal.writeln.mockReset();
      mockTerminal.reset.mockReset();
    }
  });

  it("should not call fit() when terminal.element is null", () => {
    // Temporarily set element to null to simulate terminal not being opened
    const originalElement = mockTerminal.element;
    mockTerminal.element = null as unknown as HTMLDivElement;

    renderWithProviders(<TestTerminalComponent />);

    // fit() should not be called because terminal.element is null
    expect(mockFitAddon.fit).not.toHaveBeenCalled();

    // Restore original element
    mockTerminal.element = originalElement;
  });
});
