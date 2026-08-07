import { beforeAll, describe, expect, it, vi, afterEach } from "vitest";
import { useTerminal } from "#/hooks/use-terminal";
import { Command, resetCommandStore, useCommandStore } from "#/stores/command-store";
import { renderWithProviders } from "../../test-utils";

const CONVERSATION_ID = "test-conversation-id";

// Mock useActiveConversation
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: {
      id: CONVERSATION_ID,
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
  const mockTerminal = vi.hoisted(() => ({
    loadAddon: vi.fn(),
    open: vi.fn(),
    write: vi.fn(),
    writeln: vi.fn(),
    dispose: vi.fn(),
    onData: vi.fn(() => ({ dispose: vi.fn() })),
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

        onData = mockTerminal.onData;

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
    resetCommandStore();
  });

  it("should render", () => {
    resetCommandStore(CONVERSATION_ID);
    renderWithProviders(<TestTerminalComponent />);
  });

  it("should render the commands in the terminal", () => {
    const commands: Command[] = [
      { content: "echo hello", type: "input" },
      { content: "hello", type: "output" },
    ];

    // Set commands in store before rendering to ensure they're picked up during initialization
    resetCommandStore(CONVERSATION_ID, commands);

    renderWithProviders(<TestTerminalComponent />);

    expect(mockTerminal.write).toHaveBeenCalledWith("$ ");
    expect(mockTerminal.writeln).toHaveBeenNthCalledWith(1, "echo hello");
    expect(mockTerminal.writeln).toHaveBeenNthCalledWith(2, "hello");
  });

  it("should not call fit() when terminal.element is null", () => {
    resetCommandStore(CONVERSATION_ID);
    // Temporarily set element to null to simulate terminal not being opened
    const originalElement = mockTerminal.element;
    mockTerminal.element = null as unknown as HTMLDivElement;

    renderWithProviders(<TestTerminalComponent />);

    // fit() should not be called because terminal.element is null
    expect(mockFitAddon.fit).not.toHaveBeenCalled();

    // Restore original element
    mockTerminal.element = originalElement;
  });

  it("should skip already-displayed commands when syncing from the store", () => {
    resetCommandStore(CONVERSATION_ID, [
      { content: "pwd", type: "input", alreadyDisplayed: true },
      { content: "/workspace", type: "output", alreadyDisplayed: true },
    ]);

    renderWithProviders(<TestTerminalComponent />);

    expect(mockTerminal.writeln).not.toHaveBeenCalled();
  });

  it("should submit typed commands through onSubmitCommand", async () => {
    resetCommandStore(CONVERSATION_ID);
    let onDataHandler: ((data: string) => void) | null = null;
    mockTerminal.onData.mockImplementation((handler: (data: string) => void) => {
      onDataHandler = handler;
      return { dispose: vi.fn() };
    });

    const onSubmitCommand = vi.fn(async () => {
      useCommandStore.getState().appendOutput("ok");
    });

    function InteractiveTerminal() {
      const ref = useTerminal({ onSubmitCommand });
      return <div ref={ref} />;
    }

    renderWithProviders(<InteractiveTerminal />);

    expect(onDataHandler).not.toBeNull();
    onDataHandler!("l");
    onDataHandler!("s");
    onDataHandler!("\r");

    await vi.waitFor(() => {
      expect(onSubmitCommand).toHaveBeenCalledWith("ls");
    });

    const stored = useCommandStore.getState().commands;
    expect(stored[0]).toMatchObject({
      content: "ls",
      type: "input",
      alreadyDisplayed: true,
    });
  });
});
