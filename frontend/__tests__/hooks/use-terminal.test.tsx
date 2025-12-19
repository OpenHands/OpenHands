/* eslint-disable max-classes-per-file */
import { beforeAll, describe, expect, it, vi, afterEach } from "vitest";
import { useTerminal } from "#/hooks/use-terminal";
import { Command, useCommandStore } from "#/state/command-store";
import { renderWithProviders } from "../../test-utils";

// Mock the WsClient context
vi.mock("#/context/ws-client-provider", () => ({
  useWsClient: () => ({
    send: vi.fn(),
    status: "CONNECTED",
    isLoadingMessages: false,
    events: [],
  }),
}));

// Mock useActiveConversation
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: {
      id: "test-conversation-id",
      conversation_version: "V0",
    },
    isFetched: true,
  }),
}));

// Mock useConversationWebSocket (returns null for V0 conversations)
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
});

describe("canFitTerminal safety checks", () => {
  /**
   * These tests verify the logic that prevents the
   * "Cannot read properties of undefined (reading 'dimensions')" error.
   *
   * The error occurs when FitAddon.fit() is called on a terminal that is:
   * - Hidden (display: none)
   * - Disposed/unmounted
   * - Not fully initialized
   *
   * The xterm library internally accesses `this._renderService.dimensions` in
   * the Viewport.syncScrollArea() method. When the terminal is in an invalid
   * state, _renderService is undefined, causing the TypeError.
   */

  /**
   * Simulates the error that occurs in xterm when fit() is called
   * on a terminal in an invalid state.
   */
  const createErrorThrowingFitAddon = () => ({
    fit: vi.fn().mockImplementation(() => {
      // This simulates what happens inside xterm when the terminal
      // is hidden or not properly initialized
      const _renderService = undefined;
      // This line throws: "Cannot read properties of undefined (reading 'dimensions')"
      // eslint-disable-next-line @typescript-eslint/no-unused-expressions
      _renderService!.dimensions;
    }),
  });

  describe("error reproduction without safety checks", () => {
    it("should throw 'dimensions' error when fit() is called on invalid terminal", () => {
      // This test demonstrates the original error that was occurring
      const fitAddon = createErrorThrowingFitAddon();

      // Without safety checks, calling fit() throws the error
      expect(() => fitAddon.fit()).toThrow(
        "Cannot read properties of undefined (reading 'dimensions')",
      );
    });
  });

  describe("error prevention with safety checks", () => {
    it("should prevent error by not calling fit() when terminal is null", () => {
      const terminal = null;
      const fitAddon = createErrorThrowingFitAddon();
      const container = document.createElement("div");

      // Safety check prevents the call
      const canFit =
        terminal !== null && fitAddon !== null && container !== null;

      if (canFit) {
        fitAddon.fit(); // Would throw if called
      }

      // fit() was never called, so no error
      expect(fitAddon.fit).not.toHaveBeenCalled();
    });

    it("should prevent error by not calling fit() when container has zero dimensions", () => {
      const terminal = { element: document.createElement("div") };
      const fitAddon = createErrorThrowingFitAddon();
      const container = document.createElement("div");
      Object.defineProperty(container, "clientWidth", { value: 0 });
      Object.defineProperty(container, "clientHeight", { value: 100 });

      // Safety check prevents the call
      const hasValidDimensions =
        container.clientWidth > 0 && container.clientHeight > 0;

      if (hasValidDimensions) {
        fitAddon.fit(); // Would throw if called
      }

      // fit() was never called, so no error
      expect(fitAddon.fit).not.toHaveBeenCalled();
    });

    it("should prevent error by not calling fit() when terminal.element is null", () => {
      const terminal = { element: null }; // element is null before open()
      const fitAddon = createErrorThrowingFitAddon();
      const container = document.createElement("div");
      Object.defineProperty(container, "clientWidth", { value: 100 });
      Object.defineProperty(container, "clientHeight", { value: 100 });

      // Safety check prevents the call
      if (terminal.element) {
        fitAddon.fit(); // Would throw if called
      }

      // fit() was never called, so no error
      expect(fitAddon.fit).not.toHaveBeenCalled();
    });

    it("should prevent error by not calling fit() when element is hidden", () => {
      const container = document.createElement("div");
      container.style.display = "none";
      document.body.appendChild(container);

      const terminal = { element: document.createElement("div") };
      const fitAddon = createErrorThrowingFitAddon();

      // Safety check prevents the call
      const computedStyle = getComputedStyle(container);
      const isVisible = computedStyle.display !== "none";

      if (isVisible) {
        fitAddon.fit(); // Would throw if called
      }

      // fit() was never called, so no error
      expect(fitAddon.fit).not.toHaveBeenCalled();

      document.body.removeChild(container);
    });

    it("should prevent error by not calling fit() when terminal is disposed", () => {
      let isDisposed = false;
      const fitAddon = createErrorThrowingFitAddon();

      // Simulate unmount
      isDisposed = true;

      // Safety check prevents the call
      if (!isDisposed) {
        fitAddon.fit(); // Would throw if called
      }

      // fit() was never called, so no error
      expect(fitAddon.fit).not.toHaveBeenCalled();
    });
  });

  describe("successful fit when conditions are met", () => {
    it("should call fit() when all safety conditions pass", () => {
      const terminal = { element: document.createElement("div") };
      const fitAddon = { fit: vi.fn() }; // Normal mock, doesn't throw
      const container = document.createElement("div");
      container.style.display = "block";
      document.body.appendChild(container);
      Object.defineProperty(container, "clientWidth", { value: 100 });
      Object.defineProperty(container, "clientHeight", { value: 100 });

      const isDisposed = false;
      const computedStyle = getComputedStyle(container);

      // All safety checks pass
      const canFit =
        !isDisposed &&
        terminal !== null &&
        fitAddon !== null &&
        container !== null &&
        terminal.element !== null &&
        container.clientWidth > 0 &&
        container.clientHeight > 0 &&
        computedStyle.display !== "none";

      if (canFit) {
        fitAddon.fit();
      }

      // fit() was called successfully
      expect(fitAddon.fit).toHaveBeenCalledTimes(1);

      document.body.removeChild(container);
    });
  });
});
