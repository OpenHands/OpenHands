import { beforeAll, describe, expect, it, vi, afterEach } from "vitest";
import { act } from "@testing-library/react";
import { useTerminal } from "#/hooks/use-terminal";
import { Command, useCommandStore } from "#/stores/command-store";
import { renderWithProviders } from "../../test-utils";

// Mock useActiveConversation
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: {
      id: "test-conversation-id",
    },
    isFetched: true,
  }),
}));

// Mock useConversationWebSocket
vi.mock("#/contexts/conversation-websocket-context", () => ({
  useConversationWebSocket: () => null,
}));

function TestTerminalComponent() {
  const ref = useTerminal();
  return <div ref={ref} />;
}

describe("useTerminal", () => {
  // Terminal is read-only - no longer tests user input functionality
  const mockTerminal = vi.hoisted(() => ({
    loadAddon: vi.fn(),
    open: vi.fn(),
    write: vi.fn(),
    writeln: vi.fn(),
    dispose: vi.fn(),
    element: document.createElement("div"),
  }));

  const mockFitAddon = vi.hoisted(() => ({
    fit: vi.fn(),
  }));

  beforeAll(() => {
    // mock ResizeObserver - use class for Vitest 4 constructor support
    window.ResizeObserver = class {
      observe = vi.fn();

      unobserve = vi.fn();

      disconnect = vi.fn();
    } as unknown as typeof ResizeObserver;

    // mock Terminal - use class for Vitest 4 constructor support
    vi.mock("@xterm/xterm", async (importOriginal) => ({
      ...(await importOriginal<typeof import("@xterm/xterm")>()),
      Terminal: class {
        loadAddon = mockTerminal.loadAddon;

        open = mockTerminal.open;

        write = mockTerminal.write;

        writeln = mockTerminal.writeln;

        dispose = mockTerminal.dispose;

        element = mockTerminal.element;
      },
    }));

    // mock FitAddon
    vi.mock("@xterm/addon-fit", () => ({
      FitAddon: class {
        fit = mockFitAddon.fit;
      },
    }));
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

  it("should render new commands in two concurrent instances independently", () => {
    // Arrange - two mounted instances keyed to different conversations
    renderWithProviders(<TestTerminalComponent />, {
      navigation: { conversationId: "conv-a" },
    });
    renderWithProviders(<TestTerminalComponent />, {
      navigation: { conversationId: "conv-b" },
    });

    // Act
    act(() => {
      useCommandStore.getState().appendOutput("hello");
    });

    // Assert - each instance renders the new command (2 total). With a shared
    // singleton index, whichever instance ran first advanced the index and the
    // second rendered nothing.
    const helloCalls = mockTerminal.writeln.mock.calls.filter(
      ([line]) => line === "hello",
    );
    expect(helloCalls).toHaveLength(2);
  });

  it("should preserve terminal history across unmount and remount", () => {
    // Arrange
    const commands: Command[] = [
      { content: "echo hello", type: "input" },
      { content: "hello", type: "output" },
    ];
    useCommandStore.setState({ commands });

    const { unmount } = renderWithProviders(<TestTerminalComponent />);

    expect(mockTerminal.writeln).toHaveBeenNthCalledWith(1, "echo hello");
    expect(mockTerminal.writeln).toHaveBeenNthCalledWith(2, "hello");

    // Act - unmount and remount with the same conversation id
    unmount();
    mockTerminal.writeln.mockClear();
    renderWithProviders(<TestTerminalComponent />);

    // Assert - full history is replayed on remount
    expect(mockTerminal.writeln).toHaveBeenNthCalledWith(1, "echo hello");
    expect(mockTerminal.writeln).toHaveBeenNthCalledWith(2, "hello");

    // Assert - index continuity: a new command renders exactly once
    act(() => {
      useCommandStore.getState().appendOutput("world");
    });
    const worldCalls = mockTerminal.writeln.mock.calls.filter(
      ([line]) => line === "world",
    );
    expect(worldCalls).toHaveLength(1);
  });

  it("should not corrupt a sibling instance when one instance unmounts", () => {
    // Arrange - one replayed command in each of two instances
    useCommandStore.setState({
      commands: [{ content: "c1", type: "output" }],
    });
    renderWithProviders(<TestTerminalComponent />, {
      navigation: { conversationId: "conv-a" },
    });
    const instanceB = renderWithProviders(<TestTerminalComponent />, {
      navigation: { conversationId: "conv-b" },
    });
    expect(
      mockTerminal.writeln.mock.calls.filter(([line]) => line === "c1"),
    ).toHaveLength(2);

    // Act - unmount B, then append a new command
    instanceB.unmount();
    act(() => {
      useCommandStore.getState().appendOutput("c2");
    });

    // Assert - only A renders the new command, and A does not re-render "c1"
    // (with the old shared index, B's cleanup reset it to 0 and A duplicated
    // the replayed history)
    expect(
      mockTerminal.writeln.mock.calls.filter(([line]) => line === "c2"),
    ).toHaveLength(1);
    expect(
      mockTerminal.writeln.mock.calls.filter(([line]) => line === "c1"),
    ).toHaveLength(2);
  });

  it("should reset the index when commands are cleared so re-streamed history renders", () => {
    // Arrange - a replayed command advances the index to 1
    useCommandStore.setState({
      commands: [{ content: "c1", type: "output" }],
    });
    renderWithProviders(<TestTerminalComponent />);

    // Act - clear the store (conversation switch), then stream a new command
    act(() => {
      useCommandStore.getState().clearTerminal();
    });
    act(() => {
      useCommandStore.getState().appendOutput("c2");
    });

    // Assert - the shrink guard reset the index, so the new command renders
    expect(
      mockTerminal.writeln.mock.calls.filter(([line]) => line === "c2"),
    ).toHaveLength(1);
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
