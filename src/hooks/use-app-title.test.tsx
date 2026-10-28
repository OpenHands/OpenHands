import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useParams } from "react-router";
import { useUserConversation } from "./query/use-user-conversation";
import { useAppTitle } from "./use-app-title";
import { useConversationStateStore } from "#/stores/conversation-state-store";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import {
  useCanvasExtensionsRuntime,
  type RegisteredCanvasExtensionPage,
} from "#/components/features/canvas-extensions/canvas-extensions-runtime";

const renderAppTitleHook = () =>
  renderHook(() => useAppTitle(), {
    wrapper: ({ children }) => (
      <QueryClientProvider client={new QueryClient()}>
        {children}
      </QueryClientProvider>
    ),
  });

vi.mock("./query/use-user-conversation");
vi.mock(
  "#/components/features/canvas-extensions/canvas-extensions-runtime",
  () => ({
    useCanvasExtensionsRuntime: vi.fn(),
  }),
);
vi.mock("react-router", async () => {
  const actual = await vi.importActual("react-router");
  return {
    ...actual,
    useParams: vi.fn(),
  };
});

describe("useAppTitle", () => {
  const mockUseUserConversation = vi.mocked(useUserConversation);
  const mockUseParams = vi.mocked(useParams);
  const mockUseCanvasExtensionsRuntime = vi.mocked(useCanvasExtensionsRuntime);

  const mockPages: RegisteredCanvasExtensionPage[] = [
    {
      extension: {
        name: "demo-page",
        version: "1.0.0",
        enabled: true,
        source: "github:demo/demo-page",
        installed_at: "2026-01-01T00:00:00.000Z",
        install_path: "/tmp/demo-page",
      },
      contribution: {
        id: "hello",
        path: "hello",
        title: "Hello from an extension",
      },
      mount: vi.fn(),
      href: "/extensions/demo-page/hello",
    },
    {
      extension: {
        name: "work-queue",
        version: "1.0.0",
        enabled: true,
        source: "github:demo/work-queue",
        installed_at: "2026-01-01T00:00:00.000Z",
        install_path: "/tmp/work-queue",
      },
      contribution: {
        id: "queue",
        path: "queue",
        title: "Work Queue",
      },
      mount: vi.fn(),
      href: "/extensions/work-queue/queue",
    },
  ];

  beforeEach(() => {
    // @ts-expect-error - only returning partial config for test
    mockUseUserConversation.mockReturnValue({ data: null });
    mockUseParams.mockReturnValue({});
    mockUseCanvasExtensionsRuntime.mockReturnValue({
      pages: [],
      activating: false,
      errors: new Map(),
    });
    useConversationStateStore.getState().reset();
  });

  it("returns the OSS app title outside conversations", async () => {
    const { result } = renderAppTitleHook();

    await waitFor(() => expect(result.current).toBe("OpenHands"));
  });

  it("returns the conversation title with the OSS app name", async () => {
    mockUseParams.mockReturnValue({ conversationId: "123" });
    mockUseUserConversation.mockReturnValue({
      // @ts-expect-error - only returning partial config for test
      data: { title: "My Conversation" },
    });

    const { result } = renderAppTitleHook();

    await waitFor(() =>
      expect(result.current).toBe("My Conversation | OpenHands"),
    );
  });

  it("returns the app name while conversation data is loading", async () => {
    mockUseParams.mockReturnValue({ conversationId: "123" });
    // @ts-expect-error - only returning partial config for test
    mockUseUserConversation.mockReturnValue({ data: undefined });

    const { result } = renderAppTitleHook();

    await waitFor(() => expect(result.current).toBe("OpenHands"));
  });

  it.each([
    [ExecutionStatus.RUNNING, "🟢"],
    [ExecutionStatus.FINISHED, "✅"],
    [ExecutionStatus.IDLE, "✅"],
    [ExecutionStatus.WAITING_FOR_CONFIRMATION, "✅"],
    [ExecutionStatus.PAUSED, "⚪"],
    [ExecutionStatus.ERROR, "🔴"],
    [ExecutionStatus.STUCK, "🔴"],
  ])(
    "prefixes the title with %s emoji for execution status %s",
    async (status, emoji) => {
      mockUseParams.mockReturnValue({ conversationId: "123" });
      mockUseUserConversation.mockReturnValue({
        // @ts-expect-error - only returning partial config for test
        data: { title: "My Conversation" },
      });
      useConversationStateStore.getState().setExecutionStatus("123", status);

      const { result } = renderAppTitleHook();

      await waitFor(() =>
        expect(result.current).toBe(`${emoji} My Conversation | OpenHands`),
      );
    },
  );

  it("falls back to the conversation's execution_status when the live store is empty", async () => {
    mockUseParams.mockReturnValue({ conversationId: "123" });
    mockUseUserConversation.mockReturnValue({
      // @ts-expect-error - only returning partial config for test
      data: {
        title: "My Conversation",
        execution_status: ExecutionStatus.RUNNING,
      },
    });

    const { result } = renderAppTitleHook();

    await waitFor(() =>
      expect(result.current).toBe("🟢 My Conversation | OpenHands"),
    );
  });

  it("does not add an emoji when in a conversation but execution status is unknown", async () => {
    mockUseParams.mockReturnValue({ conversationId: "123" });
    mockUseUserConversation.mockReturnValue({
      // @ts-expect-error - only returning partial config for test
      data: { title: "My Conversation" },
    });

    const { result } = renderAppTitleHook();

    await waitFor(() =>
      expect(result.current).toBe("My Conversation | OpenHands"),
    );
  });

  describe("canvas app pages", () => {
    it("returns the app page title with the OpenHands App suffix when viewing an enabled app page", async () => {
      mockUseParams.mockReturnValue({
        extensionName: "demo-page",
        "*": "hello",
      });
      mockUseCanvasExtensionsRuntime.mockReturnValue({
        pages: mockPages,
        activating: false,
        errors: new Map(),
      });

      const { result } = renderAppTitleHook();

      await waitFor(() =>
        expect(result.current).toBe("Hello from an extension - OpenHands App"),
      );
    });

    it("uses the registered title of the matching page contribution for other apps", async () => {
      mockUseParams.mockReturnValue({
        extensionName: "work-queue",
        "*": "queue",
      });
      mockUseCanvasExtensionsRuntime.mockReturnValue({
        pages: mockPages,
        activating: false,
        errors: new Map(),
      });

      const { result } = renderAppTitleHook();

      await waitFor(() =>
        expect(result.current).toBe("Work Queue - OpenHands App"),
      );
    });

    it("preserves the app page title for nested sub-paths under the contribution path", async () => {
      mockUseParams.mockReturnValue({
        extensionName: "demo-page",
        "*": "hello/nested/subpage",
      });
      mockUseCanvasExtensionsRuntime.mockReturnValue({
        pages: mockPages,
        activating: false,
        errors: new Map(),
      });

      const { result } = renderAppTitleHook();

      await waitFor(() =>
        expect(result.current).toBe("Hello from an extension - OpenHands App"),
      );
    });

    it("falls back to OpenHands for an unknown app", async () => {
      mockUseParams.mockReturnValue({
        extensionName: "non-existent-app",
        "*": "hello",
      });
      mockUseCanvasExtensionsRuntime.mockReturnValue({
        pages: mockPages,
        activating: false,
        errors: new Map(),
      });

      const { result } = renderAppTitleHook();

      await waitFor(() => expect(result.current).toBe("OpenHands"));
    });

    it("falls back to OpenHands for an unknown page of a known app", async () => {
      mockUseParams.mockReturnValue({
        extensionName: "demo-page",
        "*": "unknown-route",
      });
      mockUseCanvasExtensionsRuntime.mockReturnValue({
        pages: mockPages,
        activating: false,
        errors: new Map(),
      });

      const { result } = renderAppTitleHook();

      await waitFor(() => expect(result.current).toBe("OpenHands"));
    });

    it("falls back to OpenHands when the app is disabled", async () => {
      const disabledPages: RegisteredCanvasExtensionPage[] = [
        {
          ...mockPages[0],
          extension: {
            ...mockPages[0].extension,
            enabled: false,
          },
        },
      ];
      mockUseParams.mockReturnValue({
        extensionName: "demo-page",
        "*": "hello",
      });
      mockUseCanvasExtensionsRuntime.mockReturnValue({
        pages: disabledPages,
        activating: false,
        errors: new Map(),
      });

      const { result } = renderAppTitleHook();

      await waitFor(() => expect(result.current).toBe("OpenHands"));
    });

    it("falls back to OpenHands when the app is not in registered pages", async () => {
      mockUseParams.mockReturnValue({
        extensionName: "demo-page",
        "*": "hello",
      });
      mockUseCanvasExtensionsRuntime.mockReturnValue({
        pages: [],
        activating: false,
        errors: new Map(),
      });

      const { result } = renderAppTitleHook();

      await waitFor(() => expect(result.current).toBe("OpenHands"));
    });
  });
});
