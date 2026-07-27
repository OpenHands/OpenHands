import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useBtwInterceptor } from "#/hooks/chat/use-btw-interceptor";
import { useBtwStore } from "#/stores/btw-store";

const mockAskAgent = vi.hoisted(() =>
  vi.fn<(id: string, q: string) => Promise<{ response: string }>>(),
);

vi.mock("#/hooks/mutation/conversation-mutation-utils", () => ({
  askAgent: (id: string, q: string) => mockAskAgent(id, q),
}));

const CONV = "conv-1";
const entries = () => useBtwStore.getState().entriesByConversation[CONV] ?? [];

describe("useBtwInterceptor", () => {
  beforeEach(() => {
    useBtwStore.setState({ entriesByConversation: {} });
    mockAskAgent.mockReset();
  });

  it("falls through to onSubmit for non-/btw messages", () => {
    const onSubmit = vi.fn();
    const { result } = renderHook(() => useBtwInterceptor(CONV, onSubmit));
    act(() => result.current("hello world"));
    expect(onSubmit).toHaveBeenCalledWith("hello world");
    expect(mockAskAgent).not.toHaveBeenCalled();
    expect(entries()).toEqual([]);
  });

  it("intercepts /btw, calls askAgent, and resolves the entry", async () => {
    mockAskAgent.mockResolvedValueOnce({ response: "because" });
    const onSubmit = vi.fn();
    const { result } = renderHook(() => useBtwInterceptor(CONV, onSubmit));

    act(() => result.current("/btw why?"));

    expect(onSubmit).not.toHaveBeenCalled();
    expect(mockAskAgent).toHaveBeenCalledWith(CONV, "why?");
    expect(entries()[0]).toMatchObject({ question: "why?", status: "pending" });

    await waitFor(() => expect(entries()[0].status).toBe("done"));
    expect(entries()[0]).toMatchObject({
      response: "because",
      status: "done",
    });
  });

  it("marks the entry as error when askAgent rejects", async () => {
    mockAskAgent.mockRejectedValueOnce(new Error("boom"));
    const { result } = renderHook(() => useBtwInterceptor(CONV, vi.fn()));
    act(() => result.current("/btw why?"));
    await waitFor(() => expect(entries()[0].status).toBe("error"));
    expect(entries()[0].response).toBe("boom");
  });

  it("falls through when conversationId is null", () => {
    const onSubmit = vi.fn();
    const { result } = renderHook(() => useBtwInterceptor(null, onSubmit));
    act(() => result.current("/btw why?"));
    expect(onSubmit).toHaveBeenCalledWith("/btw why?");
    expect(mockAskAgent).not.toHaveBeenCalled();
  });
});
