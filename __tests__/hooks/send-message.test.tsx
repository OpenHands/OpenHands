import { renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useSendMessage } from "#/hooks/use-send-message";

const { useConversationWebSocketMock, useOptionalConversationIdMock } =
  vi.hoisted(() => ({
    useConversationWebSocketMock: vi.fn(),
    useOptionalConversationIdMock: vi.fn(),
  }));

vi.mock("#/contexts/conversation-websocket-context", () => ({
  useConversationWebSocket: () => useConversationWebSocketMock(),
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => useOptionalConversationIdMock(),
}));

interface HookOptions {
  sendMessage?: ReturnType<typeof vi.fn> | null;
}

function renderSendMessageHook({
  sendMessage = vi.fn().mockResolvedValue({ queued: false }),
}: HookOptions = {}) {
  useOptionalConversationIdMock.mockReturnValue({
    conversationId: "conversation-1",
  });
  useConversationWebSocketMock.mockReturnValue(
    sendMessage
      ? {
          sendMessage,
        }
      : null,
  );

  return {
    ...renderHook(() => useSendMessage()),
    sendMessage,
  };
}

afterEach(() => {
  vi.clearAllMocks();
});

describe("sending chat messages", () => {
  it("is a safe no-op outside a conversation provider", async () => {
    const { result } = renderSendMessageHook({
      sendMessage: null,
    });

    await expect(
      result.current.send({
        action: "message",
        args: { content: "Hello" },
      }),
    ).resolves.toEqual({ queued: false });
  });

  it("ignores events that are not user messages", async () => {
    const sendMessage = vi.fn();
    const { result } = renderSendMessageHook({ sendMessage });

    await expect(
      result.current.send({
        action: "change_agent_state",
        args: { content: "This is not a chat message" },
      }),
    ).resolves.toEqual({ queued: false });
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it.each([
    ["missing arguments", { action: "message" }],
    ["missing content", { action: "message", args: {} }],
    ["empty content", { action: "message", args: { content: "" } }],
  ])("ignores a message with %s", async (_description, event) => {
    const sendMessage = vi.fn();
    const { result } = renderSendMessageHook({ sendMessage });

    await expect(result.current.send(event)).resolves.toEqual({
      queued: false,
    });
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it("converts text chat input into a user WebSocket message", async () => {
    const sendMessage = vi.fn().mockResolvedValue({ queued: true });
    const { result } = renderSendMessageHook({ sendMessage });

    await expect(
      result.current.send({
        action: "message",
        args: {
          content: "Explain this code",
          file_urls: ["file:///workspace/example.ts"],
          timestamp: "2026-07-13T00:00:00.000Z",
        },
      }),
    ).resolves.toEqual({ queued: true });
    expect(sendMessage).toHaveBeenCalledOnce();
    expect(sendMessage).toHaveBeenCalledWith({
      role: "user",
      content: [{ type: "text", text: "Explain this code" }],
    });
  });

  it("does not add an image block for an empty image list", async () => {
    const sendMessage = vi.fn().mockResolvedValue({ queued: false });
    const { result } = renderSendMessageHook({ sendMessage });

    await result.current.send({
      action: "message",
      args: { content: "No images", image_urls: [] },
    });

    expect(sendMessage).toHaveBeenCalledWith({
      role: "user",
      content: [{ type: "text", text: "No images" }],
    });
  });

  it("appends image URLs after the text content", async () => {
    const imageUrls = [
      "data:image/png;base64,first",
      "data:image/png;base64,second",
    ];
    const sendMessage = vi.fn().mockResolvedValue({ queued: false });
    const { result } = renderSendMessageHook({ sendMessage });

    await expect(
      result.current.send({
        action: "message",
        args: { content: "Compare these", image_urls: imageUrls },
      }),
    ).resolves.toEqual({ queued: false });
    expect(sendMessage).toHaveBeenCalledWith({
      role: "user",
      content: [
        { type: "text", text: "Compare these" },
        { type: "image", image_urls: imageUrls },
      ],
    });
  });

  it("uses the replacement WebSocket after the active conversation changes", async () => {
    const firstSendMessage = vi.fn().mockResolvedValue({ queued: false });
    const secondSendMessage = vi.fn().mockResolvedValue({ queued: true });
    const { result, rerender } = renderSendMessageHook({
      sendMessage: firstSendMessage,
    });

    await result.current.send({
      action: "message",
      args: { content: "First conversation" },
    });

    useOptionalConversationIdMock.mockReturnValue({
      conversationId: "conversation-2",
    });
    useConversationWebSocketMock.mockReturnValue({
      sendMessage: secondSendMessage,
    });
    rerender();

    await expect(
      result.current.send({
        action: "message",
        args: { content: "Second conversation" },
      }),
    ).resolves.toEqual({ queued: true });
    expect(firstSendMessage).toHaveBeenCalledOnce();
    expect(secondSendMessage).toHaveBeenCalledWith({
      role: "user",
      content: [{ type: "text", text: "Second conversation" }],
    });
  });

  it("propagates a WebSocket delivery failure", async () => {
    const deliveryError = new Error("WebSocket closed");
    const sendMessage = vi.fn().mockRejectedValue(deliveryError);
    const { result } = renderSendMessageHook({ sendMessage });

    await expect(
      result.current.send({
        action: "message",
        args: { content: "Please deliver this" },
      }),
    ).rejects.toBe(deliveryError);
  });
});
