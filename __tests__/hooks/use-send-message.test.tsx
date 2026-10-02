import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook } from "@testing-library/react";
import { useSendMessage } from "#/hooks/use-send-message";
import { useConversationWebSocket } from "#/contexts/conversation-websocket-context";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";

vi.mock("#/contexts/conversation-websocket-context");
vi.mock("#/hooks/use-conversation-id");

describe("useSendMessage", () => {
  const mockSendMessage = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useOptionalConversationId).mockReturnValue({
      conversationId: "conv-123",
    });
    vi.mocked(useConversationWebSocket).mockReturnValue({
      sendMessage: mockSendMessage,
    } as unknown as ReturnType<typeof useConversationWebSocket>);
    mockSendMessage.mockResolvedValue({ queued: true });
  });

  it("sends an image-only message without an empty text block", async () => {
    const { result } = renderHook(() => useSendMessage());
    const res = await result.current.send({
      action: "message",
      args: {
        content: "",
        image_urls: ["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="],
      },
    });

    expect(mockSendMessage).toHaveBeenCalledTimes(1);
    expect(mockSendMessage).toHaveBeenCalledWith({
      role: "user",
      content: [
        {
          type: "image",
          image_urls: ["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="],
        },
      ],
    });
    expect(res).toEqual({ queued: true });
  });

  it("sends text and image message with text first", async () => {
    const { result } = renderHook(() => useSendMessage());
    const res = await result.current.send({
      action: "message",
      args: {
        content: "Look at this screenshot",
        image_urls: ["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="],
      },
    });

    expect(mockSendMessage).toHaveBeenCalledTimes(1);
    expect(mockSendMessage).toHaveBeenCalledWith({
      role: "user",
      content: [
        {
          type: "text",
          text: "Look at this screenshot",
        },
        {
          type: "image",
          image_urls: ["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="],
        },
      ],
    });
    expect(res).toEqual({ queued: true });
  });

  it("sends a text-only message", async () => {
    const { result } = renderHook(() => useSendMessage());
    const res = await result.current.send({
      action: "message",
      args: {
        content: "Hello agent",
      },
    });

    expect(mockSendMessage).toHaveBeenCalledTimes(1);
    expect(mockSendMessage).toHaveBeenCalledWith({
      role: "user",
      content: [
        {
          type: "text",
          text: "Hello agent",
        },
      ],
    });
    expect(res).toEqual({ queued: true });
  });

  it("ignores messages with empty text and no images", async () => {
    const { result } = renderHook(() => useSendMessage());
    const res = await result.current.send({
      action: "message",
      args: {
        content: "   ",
        image_urls: [],
      },
    });

    expect(mockSendMessage).not.toHaveBeenCalled();
    expect(res).toEqual({ queued: false });
  });

  it("returns queued: false when conversationContext is null", async () => {
    vi.mocked(useConversationWebSocket).mockReturnValue(null);
    const { result } = renderHook(() => useSendMessage());
    const res = await result.current.send({
      action: "message",
      args: {
        content: "Hello",
      },
    });

    expect(mockSendMessage).not.toHaveBeenCalled();
    expect(res).toEqual({ queued: false });
  });
});