import { waitFor } from "@testing-library/react";
import { beforeAll, describe, expect, it, vi, afterEach } from "vitest";
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

interface MockInstance {
  writes: string[];
  dispose: ReturnType<typeof vi.fn>;
}

describe("useTerminal", () => {
  // Terminal is read-only - no longer tests user input functionality.
  // open() is async, so each test waits for its instance and unmounts
  // explicitly to keep instances from bleeding across tests.
  const instances = vi.hoisted(() => [] as MockInstance[]);

  const latest = () => instances[instances.length - 1];

  beforeAll(() => {
    // mock ResizeObserver - use class for Vitest 4 constructor support
    window.ResizeObserver = class {
      observe = vi.fn();

      unobserve = vi.fn();

      disconnect = vi.fn();
    } as unknown as typeof ResizeObserver;

    // mock the rioterm engine: open() resolves to a handle whose
    // terminal records writes per instance
    vi.mock("rioterm", () => ({
      defaultTheme: { background: "#000", foreground: "#fff" },
      open: vi.fn(async () => {
        const instance: MockInstance = { writes: [], dispose: vi.fn() };
        instances.push(instance);
        return {
          terminal: {
            write: (data: string) => instance.writes.push(data),
          },
          renderer: {},
          focus: vi.fn(),
          dispose: instance.dispose,
        };
      }),
    }));
  });

  afterEach(() => {
    vi.clearAllMocks();
    instances.length = 0;
    // Reset command store between tests
    useCommandStore.setState({ commands: [] });
  });

  it("should render and hide the cursor once opened", async () => {
    const { unmount } = renderWithProviders(<TestTerminalComponent />);
    await waitFor(() => expect(latest()?.writes).toEqual(["\x1b[?25l"]));
    unmount();
  });

  it("should render the commands in the terminal", async () => {
    const commands: Command[] = [
      { content: "echo hello", type: "input" },
      { content: "hello", type: "output" },
    ];

    // Set commands in store before rendering to ensure they're picked up during initialization
    useCommandStore.setState({ commands });

    const { unmount } = renderWithProviders(<TestTerminalComponent />);

    await waitFor(() =>
      expect(latest()?.writes).toEqual([
        "\x1b[?25l",
        "$ ",
        "echo hello\r\n",
        "hello\r\n",
      ]),
    );
    unmount();
  });

  it("should render commands that arrive after initialization", async () => {
    const { unmount } = renderWithProviders(<TestTerminalComponent />);
    await waitFor(() => expect(latest()?.writes).toEqual(["\x1b[?25l"]));

    useCommandStore.setState({
      commands: [{ content: "late output", type: "output" }],
    });

    await waitFor(() =>
      expect(latest()?.writes).toContain("late output\r\n"),
    );
    unmount();
  });

  it("should dispose the terminal on unmount", async () => {
    const { unmount } = renderWithProviders(<TestTerminalComponent />);
    await waitFor(() => expect(instances).toHaveLength(1));

    unmount();
    expect(latest().dispose).toHaveBeenCalled();
  });
});
