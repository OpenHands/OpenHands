import { beforeAll, describe, expect, it, vi, afterEach } from "vitest";
import { useTerminal } from "#/hooks/use-terminal";
import type {
  ExecuteBashObservation,
  TerminalObservation,
} from "#/types/agent-server/core";
import { act } from "@testing-library/react";
import { useEventStore, type OHEvent } from "#/stores/use-event-store";
import { createOtherActionEvent } from "test-utils";
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
  const { ref } = useTerminal();
  return <div ref={ref} />;
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

function terminalEvents(kind: "ExecuteBash" | "Terminal"): OHEvent[] {
  const common = {
    command: "echo test",
    exit_code: 0,
    timeout: false,
    content: [{ type: "text" as const, text: "saved output" }],
    metadata: {
      exit_code: 0,
      pid: 1,
      username: "u",
      hostname: "h",
      working_dir: "/workspace",
      py_interpreter_path: null,
      prefix: "",
      suffix: "",
    },
  };
  const observation: ExecuteBashObservation | TerminalObservation =
    kind === "Terminal"
      ? { ...common, kind: "TerminalObservation", is_error: false }
      : { ...common, kind: "ExecuteBashObservation", error: false };
  return [
    {
      ...createOtherActionEvent("command"),
      timestamp: "2026-01-02T00:00:00Z",
      action: {
        kind: `${kind}Action` as const,
        command: "echo test",
        is_input: false,
        timeout: null,
        reset: false,
      },
    },
    {
      id: "output",
      timestamp: "2026-01-02T00:00:01Z",
      source: "environment",
      action_id: "command",
      tool_name: "terminal",
      tool_call_id: "call-1",
      observation,
    },
  ];
}

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
    useEventStore.getState().clearEvents();
  });

  it("should render", () => {
    renderWithProviders(<TestTerminalComponent />);
  });

  it.each(["ExecuteBash", "Terminal"] as const)(
    "renders saved %s commands and output without a live socket",
    (kind) => {
      useEventStore.getState().addEvents(terminalEvents(kind));

      renderWithProviders(<TestTerminalComponent />);

      expect(mockTerminal.writeln).toHaveBeenNthCalledWith(1, "echo test");
      expect(mockTerminal.writeln).toHaveBeenNthCalledWith(2, "saved output");
    },
  );

  it("appends live output once and keeps it when the terminal remounts", () => {
    const events = terminalEvents("Terminal");
    const { unmount } = renderWithProviders(<TestTerminalComponent />);

    act(() => {
      for (const event of events) useEventStore.getState().addEvent(event);
    });
    expect(mockTerminal.writeln).toHaveBeenCalledTimes(2);
    act(() => useEventStore.getState().addEvents(events));
    expect(mockTerminal.writeln).toHaveBeenCalledTimes(2);

    unmount();
    mockTerminal.writeln.mockClear();
    renderWithProviders(<TestTerminalComponent />);
    expect(mockTerminal.writeln.mock.calls.map(([text]) => text)).toEqual([
      "echo test",
      "saved output",
    ]);
  });

  it("rebuilds older pages in order and clears output on a conversation switch", () => {
    const recent = terminalEvents("Terminal");
    useEventStore.getState().addEvents(recent);
    renderWithProviders(<TestTerminalComponent />);
    mockTerminal.writeln.mockClear();
    const older = {
      ...createOtherActionEvent("older"),
      timestamp: "2026-01-01T00:00:00Z",
    };

    act(() => useEventStore.getState().addEvents([older]));

    expect(mockTerminal.reset).toHaveBeenCalledTimes(1);
    expect(mockTerminal.writeln.mock.calls.map(([text]) => text)).toEqual([
      "echo test",
      "echo test",
      "saved output",
    ]);
    act(() =>
      useEventStore
        .getState()
        .clearEventsForConversation("another-conversation"),
    );
    expect(mockTerminal.reset).toHaveBeenCalledTimes(2);
    mockTerminal.writeln.mockClear();
    act(() =>
      useEventStore.getState().addEvent({
        ...createOtherActionEvent("new-command"),
        action: {
          kind: "TerminalAction" as const,
          command: "pwd",
          is_input: false,
          timeout: null,
          reset: false,
        },
      }),
    );
    expect(mockTerminal.writeln).toHaveBeenCalledExactlyOnceWith("pwd");
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
