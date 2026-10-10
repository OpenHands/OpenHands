import { renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useSendMessage } from "#/hooks/use-send-message";

const { useConversationWebSocketMock } = vi.hoisted(() => ({
  useConversationWebSocketMock: vi.fn(),
}));

vi.mock("#/contexts/conversation-websocket-context", () => ({
  useConversationWebSocket: () => useConversationWebSocketMock(),
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: "conversation-1" }),
}));

function renderSendMessageHook(
  sendMessage: ReturnType<typeof vi.fn> | null = vi
    .fn()
    .mockResolvedValue({ queued: false }),
) {
  useConversationWebSocketMock.mockReturnValue(
    sendMessage ? { sendMessage } : null,
  );
  return renderHook(() => useSendMessage()).result;
}

afterEach(() => {
  vi.clearAllMocks();
});

describe("useSendMessage", () => {
  it("sends an image-only message without a text block", async () => {
    const sendMessage = vi.fn().mockResolvedValue({ queued: false });
    const result = renderSendMessageHook(sendMessage);

    await expect(
      result.current.send({
        action: "message",
        args: { content: "", image_urls: ["data:image/png;base64,AAAA"] },
      }),
    ).resolves.toEqual({ queued: false });

    expect(sendMessage).toHaveBeenCalledExactlyOnceWith({
      role: "user",
      content: [{ type: "image", image_urls: ["data:image/png;base64,AAAA"] }],
    });
  });

  it("preserves every image in an image-only message", async () => {
    const imageUrls = [
      "data:image/png;base64,first",
      "data:image/png;base64,second",
    ];
    const sendMessage = vi.fn().mockResolvedValue({ queued: true });
    const result = renderSendMessageHook(sendMessage);

    await expect(
      result.current.send({
        action: "message",
        args: { image_urls: imageUrls },
      }),
    ).resolves.toEqual({ queued: true });

    expect(sendMessage).toHaveBeenCalledExactlyOnceWith({
      role: "user",
      content: [{ type: "image", image_urls: imageUrls }],
    });
  });

  it("sends text followed by images", async () => {
    const sendMessage = vi.fn().mockResolvedValue({ queued: false });
    const result = renderSendMessageHook(sendMessage);

    await result.current.send({
      action: "message",
      args: { content: "look", image_urls: ["data:image/png;base64,AAAA"] },
    });

    expect(sendMessage).toHaveBeenCalledExactlyOnceWith({
      role: "user",
      content: [
        { type: "text", text: "look" },
        { type: "image", image_urls: ["data:image/png;base64,AAAA"] },
      ],
    });
  });

  it("sends a text-only message", async () => {
    const sendMessage = vi.fn().mockResolvedValue({ queued: false });
    const result = renderSendMessageHook(sendMessage);

    await result.current.send({
      action: "message",
      args: { content: "hello", image_urls: [] },
    });

    expect(sendMessage).toHaveBeenCalledExactlyOnceWith({
      role: "user",
      content: [{ type: "text", text: "hello" }],
    });
  });

  it.each([
    ["no arguments", { action: "message" }],
    ["empty text", { action: "message", args: { content: "" } }],
    [
      "empty text and no images",
      { action: "message", args: { content: "", image_urls: [] } },
    ],
  ])("does not send a message with %s", async (_description, event) => {
    const sendMessage = vi.fn();
    const result = renderSendMessageHook(sendMessage);

    await expect(result.current.send(event)).resolves.toEqual({
      queued: false,
    });
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it("propagates a delivery failure for an image-only message", async () => {
    const deliveryError = new Error("WebSocket closed");
    const sendMessage = vi.fn().mockRejectedValue(deliveryError);
    const result = renderSendMessageHook(sendMessage);

    await expect(
      result.current.send({
        action: "message",
        args: { image_urls: ["data:image/png;base64,AAAA"] },
      }),
    ).rejects.toBe(deliveryError);
  });

  it("is a no-op outside a conversation provider", async () => {
    const result = renderSendMessageHook(null);

    await expect(
      result.current.send({
        action: "message",
        args: { image_urls: ["data:image/png;base64,AAAA"] },
      }),
    ).resolves.toEqual({ queued: false });
  });
});
