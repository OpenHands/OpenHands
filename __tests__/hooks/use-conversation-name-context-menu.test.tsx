import type React from "react";
import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useConversationNameContextMenu } from "#/hooks/use-conversation-name-context-menu";
import { I18nKey } from "#/i18n/declaration";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import { useEventStore } from "#/stores/use-event-store";
import { setStoredConversationMetadata } from "#/api/conversation-metadata-store";
import type { Backend } from "#/api/backend-registry/types";

const harness = vi.hoisted(() => ({
  currentConversationId: null as string | null,
  navigate: vi.fn(),
  backend: {
    id: "local",
    name: "Local",
    host: "http://localhost:3000",
    apiKey: "",
    kind: "local" as "local" | "cloud",
  } as Backend,
  deleteConversation: vi.fn(),
  stopConversation: vi.fn(),
  updatePublicFlag: vi.fn(),
  downloadConversation: vi.fn(),
  conversation: null as { public: boolean } | null,
  systemMessage: null as {
    content: string;
    tools: null;
    openhands_version: null;
    agent_class: null;
  } | null,
  adaptSystemMessage: vi.fn(),
  displaySuccessToast: vi.fn(),
  navigateToTab: vi.fn(),
}));

vi.mock("#/hooks/use-select-conversation-tab", () => ({
  useSelectConversationTab: () => ({ navigateToTab: harness.navigateToTab }),
}));

vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => ({
    currentPath: "/",
    conversationId: harness.currentConversationId,
    isNavigating: false,
    navigate: harness.navigate,
  }),
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({ backend: harness.backend, orgId: null }),
}));

vi.mock("#/hooks/mutation/use-delete-conversation", () => ({
  useDeleteConversation: () => ({ mutate: harness.deleteConversation }),
}));

vi.mock("#/hooks/mutation/use-unified-stop-conversation", () => ({
  useUnifiedPauseConversation: () => ({ mutate: harness.stopConversation }),
}));

vi.mock("#/hooks/mutation/use-update-conversation-public-flag", () => ({
  useUpdateConversationPublicFlag: () => ({
    mutate: harness.updatePublicFlag,
  }),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({ data: harness.conversation }),
}));

vi.mock("#/hooks/use-download-conversation", () => ({
  useDownloadConversation: () => ({
    mutateAsync: harness.downloadConversation,
  }),
}));

vi.mock("#/utils/system-message-adapter", () => ({
  adaptSystemMessage: (events: unknown) => {
    harness.adaptSystemMessage(events);
    return harness.systemMessage;
  },
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: harness.displaySuccessToast,
}));

const clickEvent = () => {
  const preventDefault = vi.fn();
  const stopPropagation = vi.fn();
  return {
    event: {
      preventDefault,
      stopPropagation,
    } as unknown as React.MouseEvent<HTMLButtonElement>,
    preventDefault,
    stopPropagation,
  };
};

const pluginMetadata = (plugins: Array<{ source: string }> | null) => ({
  selected_repository: null,
  selected_branch: null,
  git_provider: null,
  plugins:
    plugins?.map(({ source }) => ({
      source,
      ref: null,
      repo_path: null,
    })) ?? null,
});

describe("useConversationNameContextMenu", () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.clearAllMocks();
    harness.currentConversationId = null;
    harness.backend = {
      id: "local",
      name: "Local",
      host: "https://local-api.example.com",
      apiKey: "",
      kind: "local",
    };
    harness.conversation = null;
    harness.systemMessage = null;
    harness.downloadConversation.mockResolvedValue(undefined);
    useEventStore.setState({ events: [], eventIds: new Set(), uiEvents: [] });
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: vi.fn().mockResolvedValue(undefined) },
    });
  });

  it("keeps actions safe when no conversation is selected", async () => {
    const events = useEventStore.getState().events;
    const { result } = renderHook(() => useConversationNameContextMenu({}));

    expect(harness.adaptSystemMessage).toHaveBeenCalledWith(events);
    expect(result.current.shareUrl).toBe("");
    expect(result.current.systemMessage).toBeNull();
    expect(result.current.systemModalVisible).toBe(false);
    expect(result.current.skillsModalVisible).toBe(false);
    expect(result.current.pluginsModalVisible).toBe(false);
    expect(result.current.hooksModalVisible).toBe(false);
    expect(result.current.confirmDeleteModalVisible).toBe(false);
    expect(result.current.confirmStopModalVisible).toBe(false);
    expect(result.current.shouldShowStop).toBe(false);
    expect(result.current.shouldShowDownloadConversation).toBe(false);
    expect(result.current.shouldShowDisplayCost).toBe(false);
    expect(result.current.shouldShowAgentTools).toBe(false);
    expect(result.current.shouldShowSkills).toBe(false);
    expect(result.current.shouldShowPlugins).toBe(false);
    expect(result.current.shouldShowHooks).toBe(false);

    const deleteClick = clickEvent();
    act(() => result.current.handleDelete(deleteClick.event));
    expect(deleteClick.preventDefault).toHaveBeenCalledOnce();
    expect(deleteClick.stopPropagation).toHaveBeenCalledOnce();
    expect(result.current.confirmDeleteModalVisible).toBe(true);
    act(() => result.current.handleConfirmDelete());
    expect(harness.deleteConversation).not.toHaveBeenCalled();
    expect(result.current.confirmDeleteModalVisible).toBe(false);

    const stopClick = clickEvent();
    act(() => result.current.handleStop(stopClick.event));
    expect(result.current.confirmStopModalVisible).toBe(true);
    act(() => result.current.handleConfirmStop());
    expect(harness.stopConversation).not.toHaveBeenCalled();
    expect(result.current.confirmStopModalVisible).toBe(false);

    const downloadClick = clickEvent();
    await act(async () =>
      result.current.handleDownloadConversation(downloadClick.event),
    );
    expect(harness.downloadConversation).not.toHaveBeenCalled();

    const editClick = clickEvent();
    act(() => result.current.handleEdit(editClick.event));
    expect(editClick.preventDefault).toHaveBeenCalledOnce();
    expect(editClick.stopPropagation).toHaveBeenCalledOnce();

    act(() => result.current.handleTogglePublic());
    expect(harness.updatePublicFlag).not.toHaveBeenCalled();

    const copyClick = clickEvent();
    act(() => result.current.handleCopyShareLink(copyClick.event));
    expect(navigator.clipboard.writeText).not.toHaveBeenCalled();
    expect(harness.displaySuccessToast).not.toHaveBeenCalled();
  });

  it("closes the menu without copying when no share link is available", () => {
    const onContextMenuToggle = vi.fn();
    const { result } = renderHook(() =>
      useConversationNameContextMenu({ onContextMenuToggle }),
    );
    const click = clickEvent();

    act(() => result.current.handleCopyShareLink(click.event));

    expect(click.preventDefault).toHaveBeenCalledOnce();
    expect(click.stopPropagation).toHaveBeenCalledOnce();
    expect(onContextMenuToggle).toHaveBeenCalledOnce();
    expect(onContextMenuToggle).toHaveBeenCalledWith(false);
    expect(navigator.clipboard.writeText).not.toHaveBeenCalled();
    expect(harness.displaySuccessToast).not.toHaveBeenCalled();
  });

  it("requires both an id and conversation before toggling public access", () => {
    const first = renderHook(() =>
      useConversationNameContextMenu({ conversationId: "conv-missing-data" }),
    );

    act(() => first.result.current.handleTogglePublic(false));
    expect(harness.updatePublicFlag).not.toHaveBeenCalled();
    first.unmount();

    harness.conversation = { public: false };
    const second = renderHook(() => useConversationNameContextMenu({}));

    act(() => second.result.current.handleTogglePublic(false));
    expect(harness.updatePublicFlag).not.toHaveBeenCalled();
  });

  it("opens every modal and closes the context menu for visible actions", () => {
    const onContextMenuToggle = vi.fn();
    const { result } = renderHook(() =>
      useConversationNameContextMenu({
        conversationId: "conv-actions",
        showOptions: true,
        onContextMenuToggle,
      }),
    );

    const cases = [
      ["handleShowAgentTools", "systemModalVisible"],
      ["handleShowSkills", "skillsModalVisible"],
      ["handleShowPlugins", "pluginsModalVisible"],
      ["handleShowHooks", "hooksModalVisible"],
    ] as const;
    for (const [handler, visible] of cases) {
      const click = clickEvent();
      onContextMenuToggle.mockClear();
      act(() => result.current[handler](click.event));
      expect(click.stopPropagation).toHaveBeenCalledOnce();
      expect(result.current[visible]).toBe(true);
      expect(onContextMenuToggle).toHaveBeenCalledOnce();
      expect(onContextMenuToggle).toHaveBeenCalledWith(false);
    }

    // Displaying cost now navigates to the usage tab and closes the menu
    // instead of opening a modal.
    const costClick = clickEvent();
    onContextMenuToggle.mockClear();
    act(() => result.current.handleDisplayCost(costClick.event));
    expect(costClick.stopPropagation).toHaveBeenCalledOnce();
    expect(harness.navigateToTab).toHaveBeenCalledWith("usage");
    expect(onContextMenuToggle).toHaveBeenCalledWith(false);

    act(() => {
      result.current.setSystemModalVisible(false);
      result.current.setSkillsModalVisible(false);
      result.current.setPluginsModalVisible(false);
      result.current.setHooksModalVisible(false);
    });
    expect(result.current.systemModalVisible).toBe(false);
    expect(result.current.skillsModalVisible).toBe(false);
    expect(result.current.pluginsModalVisible).toBe(false);
    expect(result.current.hooksModalVisible).toBe(false);
  });

  it("opens modal actions safely without a context-menu callback", () => {
    const { result } = renderHook(() =>
      useConversationNameContextMenu({
        conversationId: "conv-actions",
        showOptions: true,
      }),
    );

    const cases = [
      ["handleShowAgentTools", "systemModalVisible"],
      ["handleShowSkills", "skillsModalVisible"],
      ["handleShowPlugins", "pluginsModalVisible"],
      ["handleShowHooks", "hooksModalVisible"],
    ] as const;
    for (const [handler, visible] of cases) {
      act(() => result.current[handler](clickEvent().event));
      expect(result.current[visible]).toBe(true);
    }

    // Cost display navigates without a modal even when no toggle callback is
    // provided.
    act(() => result.current.handleDisplayCost(clickEvent().event));
    expect(harness.navigateToTab).toHaveBeenCalledWith("usage");
  });

  it("downloads, stops, shares, toggles public access, and deletes an active conversation", async () => {
    harness.currentConversationId = "conv-active";
    harness.conversation = { public: false };
    harness.systemMessage = {
      content: "system prompt",
      tools: null,
      openhands_version: null,
      agent_class: null,
    };
    setStoredConversationMetadata(
      "conv-active",
      pluginMetadata([{ source: "github:acme/plugin" }]),
    );
    const onContextMenuToggle = vi.fn();
    const { result } = renderHook(() =>
      useConversationNameContextMenu({
        conversationId: "conv-active",
        executionStatus: ExecutionStatus.RUNNING,
        showOptions: true,
        onContextMenuToggle,
      }),
    );

    expect(result.current.shouldShowStop).toBe(true);
    expect(result.current.shouldShowDownloadConversation).toBe(true);
    expect(result.current.shouldShowDisplayCost).toBe(true);
    expect(result.current.shouldShowAgentTools).toBe(true);
    expect(result.current.shouldShowSkills).toBe(true);
    expect(result.current.shouldShowPlugins).toBe(true);
    expect(result.current.shouldShowHooks).toBe(true);
    expect(result.current.shareUrl).toBe(
      `${window.location.origin}/shared/conversations/conv-active`,
    );

    const downloadClick = clickEvent();
    onContextMenuToggle.mockClear();
    await act(async () =>
      result.current.handleDownloadConversation(downloadClick.event),
    );
    expect(harness.downloadConversation).toHaveBeenCalledWith("conv-active");
    expect(onContextMenuToggle).toHaveBeenCalledOnce();
    expect(onContextMenuToggle).toHaveBeenCalledWith(false);

    const editClick = clickEvent();
    onContextMenuToggle.mockClear();
    act(() => result.current.handleEdit(editClick.event));
    expect(editClick.preventDefault).toHaveBeenCalledOnce();
    expect(editClick.stopPropagation).toHaveBeenCalledOnce();
    expect(onContextMenuToggle).toHaveBeenCalledOnce();
    expect(onContextMenuToggle).toHaveBeenCalledWith(false);

    act(() => result.current.handleTogglePublic());
    act(() => result.current.handleTogglePublic(false));
    expect(harness.updatePublicFlag).toHaveBeenNthCalledWith(1, {
      conversationId: "conv-active",
      isPublic: true,
    });
    expect(harness.updatePublicFlag).toHaveBeenNthCalledWith(2, {
      conversationId: "conv-active",
      isPublic: false,
    });

    const copyClick = clickEvent();
    act(() => result.current.handleCopyShareLink(copyClick.event));
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
      `${window.location.origin}/shared/conversations/conv-active`,
    );
    expect(harness.displaySuccessToast).toHaveBeenCalledWith(
      I18nKey.CONVERSATION$LINK_COPIED,
    );

    const stopClick = clickEvent();
    onContextMenuToggle.mockClear();
    act(() => result.current.handleStop(stopClick.event));
    expect(onContextMenuToggle).toHaveBeenCalledOnce();
    expect(onContextMenuToggle).toHaveBeenCalledWith(false);
    act(() => result.current.handleConfirmStop());
    expect(harness.stopConversation).toHaveBeenCalledWith({
      conversationId: "conv-active",
    });

    const deleteClick = clickEvent();
    onContextMenuToggle.mockClear();
    act(() => result.current.handleDelete(deleteClick.event));
    expect(onContextMenuToggle).toHaveBeenCalledOnce();
    expect(onContextMenuToggle).toHaveBeenCalledWith(false);
    act(() => result.current.handleConfirmDelete());
    expect(harness.deleteConversation).toHaveBeenCalledWith(
      { conversationId: "conv-active" },
      expect.objectContaining({ onSuccess: expect.any(Function) }),
    );
    const options = harness.deleteConversation.mock.calls[0][1] as {
      onSuccess: () => void;
    };
    act(() => options.onSuccess());
    expect(harness.navigate).toHaveBeenCalledWith("/conversations");
  });

  it("does not navigate after deleting a conversation that is not currently open", () => {
    harness.currentConversationId = "conv-current";
    const { result } = renderHook(() =>
      useConversationNameContextMenu({ conversationId: "conv-other" }),
    );

    act(() => result.current.handleConfirmDelete());
    const options = harness.deleteConversation.mock.calls[0][1] as {
      onSuccess: () => void;
    };
    act(() => options.onSuccess());

    expect(harness.navigate).not.toHaveBeenCalled();
  });

  it("updates share links when the conversation or backend changes", () => {
    const props = { conversationId: "conv-local" };
    const { result, rerender } = renderHook(() =>
      useConversationNameContextMenu(props),
    );

    expect(result.current.shareUrl).toBe(
      `${window.location.origin}/shared/conversations/conv-local`,
    );

    props.conversationId = "conv-next";
    rerender();
    expect(result.current.shareUrl).toBe(
      `${window.location.origin}/shared/conversations/conv-next`,
    );

    harness.backend = {
      id: "cloud",
      name: "Cloud",
      host: "https://app.example.com////",
      apiKey: "token",
      kind: "cloud",
    };
    rerender();
    expect(result.current.shareUrl).toBe(
      "https://app.example.com/shared/conversations/conv-next",
    );
  });

  it("keeps populated agent metadata hidden while options are closed", () => {
    harness.systemMessage = {
      content: "system",
      tools: null,
      openhands_version: null,
      agent_class: null,
    };
    setStoredConversationMetadata(
      "conv-hidden-options",
      pluginMetadata([{ source: "github:acme/plugin" }]),
    );

    const { result } = renderHook(() =>
      useConversationNameContextMenu({
        conversationId: "conv-hidden-options",
        executionStatus: ExecutionStatus.RUNNING,
        showOptions: false,
      }),
    );

    expect(result.current.shouldShowAgentTools).toBe(false);
    expect(result.current.shouldShowPlugins).toBe(false);
    expect(result.current.shouldShowHooks).toBe(false);
  });

  it("updates visibility as options, metadata, and execution status change", () => {
    setStoredConversationMetadata("conv-visibility", pluginMetadata(null));
    const props = {
      conversationId: "conv-visibility" as string | undefined,
      executionStatus: ExecutionStatus.FINISHED as ExecutionStatus | null,
      showOptions: false,
    };
    const { result, rerender } = renderHook(() =>
      useConversationNameContextMenu(props),
    );

    expect(result.current.shouldShowDownloadConversation).toBe(false);
    expect(result.current.shouldShowAgentTools).toBe(false);
    expect(result.current.shouldShowPlugins).toBe(false);
    expect(result.current.shouldShowHooks).toBe(false);

    props.showOptions = true;
    rerender();
    expect(result.current.shouldShowDownloadConversation).toBe(true);
    expect(result.current.shouldShowSkills).toBe(true);
    expect(result.current.shouldShowPlugins).toBe(false);

    setStoredConversationMetadata("conv-visibility", pluginMetadata([]));
    rerender();
    expect(result.current.shouldShowPlugins).toBe(false);

    setStoredConversationMetadata(
      "conv-visibility",
      pluginMetadata([{ source: "github:acme/plugin" }]),
    );
    props.executionStatus = ExecutionStatus.PAUSED;
    harness.systemMessage = {
      content: "system",
      tools: null,
      openhands_version: null,
      agent_class: null,
    };
    rerender();
    expect(result.current.shouldShowPlugins).toBe(true);
    expect(result.current.shouldShowAgentTools).toBe(true);
    expect(result.current.shouldShowStop).toBe(false);
    expect(result.current.shouldShowHooks).toBe(false);

    props.conversationId = undefined;
    rerender();
    expect(result.current.shouldShowDownloadConversation).toBe(false);
    expect(result.current.shouldShowSkills).toBe(false);
    expect(result.current.shouldShowPlugins).toBe(false);
  });
});
