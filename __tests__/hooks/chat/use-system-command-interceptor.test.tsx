import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useSystemCommandInterceptor } from "#/hooks/chat/use-system-command-interceptor";
import { useSlashCommandOutputStore } from "#/stores/slash-command-output-store";
import { useEventStore } from "#/stores/use-event-store";

const {
  mockCondenseConversation,
  mockDisplayErrorToast,
  mockDisplaySuccessToast,
  mockRefetchHooks,
  mockRefetchSettings,
  mockRefetchSkills,
} = vi.hoisted(() => ({
  mockCondenseConversation: vi.fn(),
  mockDisplayErrorToast: vi.fn(),
  mockDisplaySuccessToast: vi.fn(),
  mockRefetchHooks: vi.fn(),
  mockRefetchSettings: vi.fn(),
  mockRefetchSkills: vi.fn(),
}));

const skills = [
  {
    name: "code-search",
    type: "agentskills" as const,
    source: "project",
    description: "Search the current workspace",
    content: "Search the current workspace",
    triggers: ["/code-search"],
  },
];

const hooks = [
  {
    event_type: "pre_tool_use",
    matchers: [
      {
        matcher: "terminal",
        hooks: [{ type: "command", command: "npm test", timeout: 30 }],
      },
    ],
  },
];

const settings = {
  mcp_config: {
    sse_servers: [],
    stdio_servers: [
      { name: "filesystem", command: "npx", args: ["mcp-filesystem"] },
      { name: "disabled", command: "npx", enabled: false },
    ],
    shttp_servers: [],
  },
};

vi.mock("#/hooks/query/use-conversation-skills", () => ({
  useConversationSkills: () => ({
    data: skills,
    isLoading: false,
    refetch: mockRefetchSkills,
  }),
}));

vi.mock("#/hooks/query/use-conversation-hooks", () => ({
  useConversationHooks: () => ({
    data: hooks,
    isLoading: false,
    refetch: mockRefetchHooks,
  }),
}));

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => ({
    data: settings,
    isLoading: false,
    refetch: mockRefetchSettings,
  }),
}));

vi.mock("#/hooks/mutation/conversation-mutation-utils", () => ({
  condenseConversation: mockCondenseConversation,
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: mockDisplayErrorToast,
  displaySuccessToast: mockDisplaySuccessToast,
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

const CONVERSATION_ID = "conv-1";

const makeUserMessage = (id: string) => ({
  id,
  timestamp: "2026-08-02T00:00:00.000Z",
  source: "user" as const,
  llm_message: {
    role: "user" as const,
    content: [{ type: "text" as const, text: "hello" }],
  },
  activated_microagents: [],
  extended_content: [],
});

describe("useSystemCommandInterceptor", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockRefetchSkills.mockResolvedValue({ data: skills, isError: false });
    mockRefetchHooks.mockResolvedValue({ data: hooks, isError: false });
    mockRefetchSettings.mockResolvedValue({ data: settings, isError: false });
    mockCondenseConversation.mockResolvedValue(undefined);
    useEventStore.getState().clearEvents();
    useSlashCommandOutputStore.getState().clearAll();
  });

  // @spec SC-002 — Inline help
  it("renders /help from the same built-in and skill-derived command list", async () => {
    useEventStore.getState().addEvent(makeUserMessage("message-1"));
    const onSubmit = vi.fn();
    const { result } = renderHook(() =>
      useSystemCommandInterceptor(CONVERSATION_ID, onSubmit),
    );

    act(() => result.current(" /help "));

    await waitFor(() => {
      const entry =
        useSlashCommandOutputStore.getState().entriesByConversation[
          CONVERSATION_ID
        ]?.[0];
      expect(entry).toMatchObject({
        kind: "help",
        anchorEventId: "message-1",
      });
      if (entry?.kind === "help") {
        expect(entry.commands.map((command) => command.command)).toEqual(
          expect.arrayContaining(["/help", "/condense", "/code-search"]),
        );
      }
    });
    expect(onSubmit).not.toHaveBeenCalled();
  });

  // @spec SC-002 — Inline help
  it("still renders built-in help when refreshing skills fails", async () => {
    mockRefetchSkills.mockResolvedValueOnce({
      data: undefined,
      isError: true,
      error: new Error("skills unavailable"),
    });
    const onSubmit = vi.fn();
    const { result } = renderHook(() =>
      useSystemCommandInterceptor(CONVERSATION_ID, onSubmit),
    );

    act(() => result.current("/help"));

    await waitFor(() => {
      const entry =
        useSlashCommandOutputStore.getState().entriesByConversation[
          CONVERSATION_ID
        ]?.[0];
      expect(entry).toMatchObject({ kind: "help" });
      if (entry?.kind === "help") {
        expect(entry.commands.map((command) => command.command)).toContain(
          "/help",
        );
      }
    });
  });

  // @spec SC-003 — Loaded extensions
  it("renders workspace skills, conversation hooks, and enabled MCP servers for /skills", async () => {
    const onSubmit = vi.fn();
    const { result } = renderHook(() =>
      useSystemCommandInterceptor(CONVERSATION_ID, onSubmit),
    );

    act(() => result.current("/skills"));

    await waitFor(() => {
      expect(
        useSlashCommandOutputStore.getState().entriesByConversation[
          CONVERSATION_ID
        ]?.[0],
      ).toMatchObject({
        kind: "skills",
        skills,
        hooks,
        mcpServers: [
          expect.objectContaining({
            type: "stdio",
            name: "filesystem",
            command: "npx",
          }),
        ],
      });
    });
    expect(onSubmit).not.toHaveBeenCalled();
  });

  // @spec SC-003 — Loaded extensions
  it("renders available skills when hooks and settings refreshes fail", async () => {
    mockRefetchHooks.mockResolvedValueOnce({
      data: undefined,
      isError: true,
      error: new Error("hooks unavailable"),
    });
    mockRefetchSettings.mockResolvedValueOnce({
      data: undefined,
      isError: true,
      error: new Error("settings unavailable"),
    });
    const onSubmit = vi.fn();
    const { result } = renderHook(() =>
      useSystemCommandInterceptor(CONVERSATION_ID, onSubmit),
    );

    act(() => result.current("/skills"));

    await waitFor(() => {
      expect(
        useSlashCommandOutputStore.getState().entriesByConversation[
          CONVERSATION_ID
        ]?.[0],
      ).toMatchObject({ kind: "skills", skills });
    });
  });

  // @spec SC-004 — Feedback
  it("opens the anonymous feedback form in a protected new tab", () => {
    const open = vi.spyOn(window, "open").mockImplementation(() => null);
    const onSubmit = vi.fn();
    const { result } = renderHook(() =>
      useSystemCommandInterceptor(CONVERSATION_ID, onSubmit),
    );

    act(() => result.current("/feedback"));

    expect(open).toHaveBeenCalledWith(
      "https://forms.gle/chHc5VdS3wty5DwW6",
      "_blank",
      "noopener,noreferrer",
    );
    expect(onSubmit).not.toHaveBeenCalled();
  });

  // @spec SC-005 — Conversation condensation
  it("condenses the active conversation and reports success", async () => {
    const onSubmit = vi.fn();
    const { result } = renderHook(() =>
      useSystemCommandInterceptor(CONVERSATION_ID, onSubmit),
    );

    act(() => result.current("/condense"));

    await waitFor(() => {
      expect(mockCondenseConversation).toHaveBeenCalledWith(CONVERSATION_ID);
      expect(mockDisplaySuccessToast).toHaveBeenCalledWith(
        "SLASH_COMMAND$CONDENSE_SUCCESS",
      );
    });
    expect(onSubmit).not.toHaveBeenCalled();
  });

  // @spec SC-005 — Conversation condensation
  it.each([404, 405, 501])(
    "reports unsupported condensation for an HTTP %s response",
    async (status) => {
      mockCondenseConversation.mockRejectedValueOnce({ response: { status } });
      const onSubmit = vi.fn();
      const { result } = renderHook(() =>
        useSystemCommandInterceptor(CONVERSATION_ID, onSubmit),
      );

      act(() => result.current("/condense"));

      await waitFor(() => {
        expect(mockDisplayErrorToast).toHaveBeenCalledWith(
          "SLASH_COMMAND$CONDENSE_UNSUPPORTED",
        );
      });
    },
  );

  // @spec SC-005 — Conversation condensation
  it("uses the localized generic message for other condensation failures", async () => {
    mockCondenseConversation.mockRejectedValueOnce(new Error("private detail"));
    const onSubmit = vi.fn();
    const { result } = renderHook(() =>
      useSystemCommandInterceptor(CONVERSATION_ID, onSubmit),
    );

    act(() => result.current("/condense"));

    await waitFor(() => {
      expect(mockDisplayErrorToast).toHaveBeenCalledWith(
        "SLASH_COMMAND$CONDENSE_FAILED",
      );
    });
  });

  it("passes ordinary messages through unchanged", () => {
    const onSubmit = vi.fn();
    const { result } = renderHook(() =>
      useSystemCommandInterceptor(CONVERSATION_ID, onSubmit),
    );

    act(() => result.current("please run /help later"));

    expect(onSubmit).toHaveBeenCalledWith("please run /help later");
  });
});
