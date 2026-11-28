import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useCondenseInterceptor } from "./use-condense-interceptor";

const condenseConversation = vi.fn();
const displayErrorToast = vi.fn();
const displaySuccessToast = vi.fn();

vi.mock("#/hooks/mutation/conversation-mutation-utils", () => ({
  condenseConversation: (...args: unknown[]) => condenseConversation(...args),
}));
vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: (...args: unknown[]) => displayErrorToast(...args),
  displaySuccessToast: (...args: unknown[]) => displaySuccessToast(...args),
}));

const CONV = "conv-1";

const setup = (conversationId: string | null) => {
  const onSubmit = vi.fn();
  const { result } = renderHook(() =>
    useCondenseInterceptor(conversationId, onSubmit),
  );
  return { intercept: result.current, onSubmit };
};

describe("useCondenseInterceptor", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    condenseConversation.mockResolvedValue(undefined);
  });

  it("passes a non-condense message straight through to onSubmit", () => {
    const { intercept, onSubmit } = setup(CONV);
    intercept("hello there");
    expect(onSubmit).toHaveBeenCalledWith("hello there");
    expect(condenseConversation).not.toHaveBeenCalled();
  });

  it("condenses the conversation and toasts success", async () => {
    const { intercept, onSubmit } = setup(CONV);
    intercept("/condense");
    expect(condenseConversation).toHaveBeenCalledWith(CONV);
    expect(onSubmit).not.toHaveBeenCalled();
    await vi.waitFor(() =>
      expect(displaySuccessToast).toHaveBeenCalledWith(
        "Conversation history condensed.",
      ),
    );
    expect(displayErrorToast).not.toHaveBeenCalled();
  });

  it("toasts the backend error when condensation fails", async () => {
    condenseConversation.mockRejectedValue(new Error("condense boom"));
    const { intercept } = setup(CONV);
    intercept("/condense");
    await vi.waitFor(() =>
      expect(displayErrorToast).toHaveBeenCalledWith("condense boom"),
    );
    expect(displaySuccessToast).not.toHaveBeenCalled();
  });

  it("passes through (no call, no toast) when there is no conversation", () => {
    const { intercept, onSubmit } = setup(null);
    intercept("/condense");
    expect(onSubmit).toHaveBeenCalledWith("/condense");
    expect(condenseConversation).not.toHaveBeenCalled();
    expect(displayErrorToast).not.toHaveBeenCalled();
    expect(displaySuccessToast).not.toHaveBeenCalled();
  });
});
