import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, onTestFinished } from "vitest";
import { renderWithProviders } from "test-utils";
import type { MessageEvent } from "#/types/agent-server/core";
import { I18nKey } from "#/i18n/declaration";
import { UserAssistantEventMessage } from "#/components/conversation-events/chat/event-message-components/user-assistant-event-message";
import { CustomChatInput } from "#/components/features/chat/custom-chat-input";
import { getConversationState } from "#/utils/conversation-local-storage";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import type { DirectConversationInfo } from "#/api/agent-server-adapter";

const { useActiveBackendMock, useOptionalConversationIdMock, navigateMock } =
  vi.hoisted(() => ({
    useActiveBackendMock: vi.fn(),
    useOptionalConversationIdMock: vi.fn(),
    navigateMock: vi.fn(),
  }));

// These provide test context (backend kind, conversation id, navigation); the
// fork behaviour is exercised through the real hook against a mocked service
// (per the repo convention: mock the service, not the hook).
vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => useOptionalConversationIdMock(),
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => useActiveBackendMock(),
}));

vi.mock("#/context/navigation-context", async (importActual) => ({
  ...(await importActual<object>()),
  useNavigation: () => ({ navigate: navigateMock }),
}));

// test-utils re-inits i18n with empty resources, so `t()` returns the key —
// which becomes the button's accessible name (aria-label).
const BRANCH_LABEL = I18nKey.CHAT_INTERFACE$BRANCH_FROM_HERE;
const forkResult = { id: "fork-123" } as DirectConversationInfo;

let forkSpy: ReturnType<typeof vi.spyOn>;
let parentSpy: ReturnType<typeof vi.spyOn>;

const makeEvent = (source: "user" | "agent", id: string): MessageEvent =>
  ({
    id,
    source,
    timestamp: "2024-01-01T00:00:00.000Z",
    llm_message: {
      role: source === "user" ? "user" : "assistant",
      content: [{ type: "text", text: "Hello world" }],
    },
    critic_result: null,
  }) as unknown as MessageEvent;

// A user message with only an image (no text) — parses to an empty string.
const makeImageOnlyEvent = (id: string): MessageEvent =>
  ({
    id,
    source: "user",
    timestamp: "2024-01-01T00:00:00.000Z",
    llm_message: {
      role: "user",
      content: [{ type: "image", image_urls: ["data:image/png;base64,AAAA"] }],
    },
    critic_result: null,
  }) as unknown as MessageEvent;

const renderMessage = (event: MessageEvent) =>
  renderWithProviders(
    <UserAssistantEventMessage
      event={event}
      isLastMessage={false}
      isFromPlanningAgent={false}
    />,
  );

// The fork's saved composer draft, which its composer restores on open.
const forkDraft = () => getConversationState(forkResult.id).draftMessage;

describe("UserAssistantEventMessage — branch action", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    useActiveBackendMock.mockReset();
    useOptionalConversationIdMock.mockReset();
    navigateMock.mockReset();
    localStorage.clear();

    useActiveBackendMock.mockReturnValue({
      backend: { kind: "local" },
      orgId: null,
    });
    useOptionalConversationIdMock.mockReturnValue({ conversationId: "conv-1" });

    ConversationService.setCurrentConversation(null);

    forkSpy = vi
      .spyOn(AgentServerConversationService, "forkConversation")
      // Modern backend (>= 1.31.0): the fork's HEAD (leaf_event_id) is exactly
      // the requested branch point, confirming the message was excluded.
      .mockImplementation((_source, fromEventId) =>
        Promise.resolve({
          ...forkResult,
          leaf_event_id: fromEventId,
        } as unknown as DirectConversationInfo),
      );
    // Default: the message has a parent (the common case), so edit-mode
    // branches before it.
    parentSpy = vi
      .spyOn(AgentServerConversationService, "getEventParentId")
      .mockResolvedValue("evt-parent");
  });

  it("branches an assistant message inclusively, without a parent lookup or prefill", async () => {
    renderMessage(makeEvent("agent", "evt-agent"));

    // Actions only appear on hover.
    fireEvent.mouseEnter(screen.getByTestId("agent-message"));
    fireEvent.click(screen.getByRole("button", { name: BRANCH_LABEL }));

    await waitFor(() =>
      expect(navigateMock).toHaveBeenCalledWith("/conversations/fork-123"),
    );
    expect(parentSpy).not.toHaveBeenCalled();
    expect(forkSpy).toHaveBeenCalledWith("conv-1", "evt-agent", undefined);
    expect(forkDraft()).toBeNull();
  });

  it("edits a user message: branches at its parent and the fork's composer opens with its text", async () => {
    // jsdom has no innerText, which the composer reads its text from.
    Object.defineProperty(HTMLElement.prototype, "innerText", {
      configurable: true,
      get(this: HTMLElement) {
        return this.textContent ?? "";
      },
    });
    onTestFinished(() => {
      Reflect.deleteProperty(HTMLElement.prototype, "innerText");
    });
    const onSubmit = vi.fn();
    // The real composer stays mounted across the route change, as in the app.
    const page = () => (
      <>
        <UserAssistantEventMessage
          event={makeEvent("user", "evt-user")}
          isLastMessage={false}
          isFromPlanningAgent={false}
        />
        <CustomChatInput onSubmit={onSubmit} />
      </>
    );
    const { rerender } = renderWithProviders(page());

    fireEvent.mouseEnter(screen.getByTestId("user-message"));
    fireEvent.click(screen.getByRole("button", { name: BRANCH_LABEL }));

    await waitFor(() =>
      expect(navigateMock).toHaveBeenCalledWith("/conversations/fork-123"),
    );
    expect(parentSpy).toHaveBeenCalledWith("conv-1", "evt-user");
    expect(forkSpy).toHaveBeenCalledWith("conv-1", "evt-parent", undefined);

    // The router commits the new route after the click handler's own tasks.
    await act(() => new Promise((resolve) => setTimeout(resolve, 0)));
    useOptionalConversationIdMock.mockReturnValue({
      conversationId: "fork-123",
    });
    rerender(page());

    await waitFor(() =>
      expect(screen.getByTestId("chat-input").textContent).toBe("Hello world"),
    );
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("does not prefill the composer when the message has no parent (inclusive fallback)", async () => {
    parentSpy.mockResolvedValue(undefined);
    renderMessage(makeEvent("user", "evt-user"));

    fireEvent.mouseEnter(screen.getByTestId("user-message"));
    fireEvent.click(screen.getByRole("button", { name: BRANCH_LABEL }));

    await waitFor(() =>
      expect(navigateMock).toHaveBeenCalledWith("/conversations/fork-123"),
    );
    expect(forkSpy).toHaveBeenCalledWith("conv-1", "evt-user", undefined);
    expect(forkDraft()).toBeNull();
  });

  it("titles the fork distinctly from its source conversation", async () => {
    ConversationService.setCurrentConversation({
      id: "conv-1",
      title: "Trip planning",
    } as AppConversation);
    renderMessage(makeEvent("user", "evt-user"));

    fireEvent.mouseEnter(screen.getByTestId("user-message"));
    fireEvent.click(screen.getByRole("button", { name: BRANCH_LABEL }));

    await waitFor(() =>
      expect(forkSpy).toHaveBeenCalledWith(
        "conv-1",
        "evt-parent",
        "Trip planning (branch)",
      ),
    );
  });

  it("does not prefill when the backend ignored from_event_id (older agent-server)", async () => {
    // leaf_event_id != requested branch point => the message was NOT excluded
    // (the fork copied the whole conversation), so prefilling would duplicate.
    forkSpy.mockImplementation(() =>
      Promise.resolve({
        ...forkResult,
        leaf_event_id: "some-other-leaf",
      } as unknown as DirectConversationInfo),
    );
    renderMessage(makeEvent("user", "evt-user"));

    fireEvent.mouseEnter(screen.getByTestId("user-message"));
    fireEvent.click(screen.getByRole("button", { name: BRANCH_LABEL }));

    await waitFor(() =>
      expect(navigateMock).toHaveBeenCalledWith("/conversations/fork-123"),
    );
    expect(forkDraft()).toBeNull();
  });

  it("branches an image-only user message inclusively (keeps the image, no prefill)", async () => {
    renderMessage(makeImageOnlyEvent("evt-img"));

    fireEvent.mouseEnter(screen.getByTestId("user-message"));
    fireEvent.click(screen.getByRole("button", { name: BRANCH_LABEL }));

    // No text to edit → branch at the message (inclusive), no parent lookup,
    // no prefill, so the image is not dropped.
    await waitFor(() =>
      expect(navigateMock).toHaveBeenCalledWith("/conversations/fork-123"),
    );
    expect(forkSpy).toHaveBeenCalledWith("conv-1", "evt-img", undefined);
    expect(parentSpy).not.toHaveBeenCalled();
    expect(forkDraft()).toBeNull();
  });

  it("omits the fork title when the tracked conversation is a different one", async () => {
    // Stale/other conversation in the shared singleton — must not be used.
    ConversationService.setCurrentConversation({
      id: "other-conv",
      title: "Stale title",
    } as AppConversation);
    renderMessage(makeEvent("user", "evt-user"));

    fireEvent.mouseEnter(screen.getByTestId("user-message"));
    fireEvent.click(screen.getByRole("button", { name: BRANCH_LABEL }));

    await waitFor(() =>
      expect(forkSpy).toHaveBeenCalledWith("conv-1", "evt-parent", undefined),
    );
  });

  it("ignores a second click while a fork is already in flight", async () => {
    renderMessage(makeEvent("user", "evt-user"));

    fireEvent.mouseEnter(screen.getByTestId("user-message"));
    const button = screen.getByRole("button", { name: BRANCH_LABEL });
    fireEvent.click(button);
    fireEvent.click(button);

    await waitFor(() => expect(forkSpy).toHaveBeenCalled());
    expect(forkSpy).toHaveBeenCalledTimes(1);
  });

  it("hides the branch action on the cloud backend", () => {
    useActiveBackendMock.mockReturnValue({
      backend: { kind: "cloud" },
      orgId: null,
    });

    renderMessage(makeEvent("user", "evt-user"));

    fireEvent.mouseEnter(screen.getByTestId("user-message"));
    expect(
      screen.queryByRole("button", { name: BRANCH_LABEL }),
    ).not.toBeInTheDocument();
  });

  it("hides the branch action outside of a conversation", () => {
    useOptionalConversationIdMock.mockReturnValue({
      conversationId: undefined,
    });

    renderMessage(makeEvent("agent", "evt-agent"));

    fireEvent.mouseEnter(screen.getByTestId("agent-message"));
    expect(
      screen.queryByRole("button", { name: BRANCH_LABEL }),
    ).not.toBeInTheDocument();
  });
});
