import { renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  FEEDBACK_FORM_URL,
  useFeedbackInterceptor,
} from "./use-feedback-interceptor";

const CONV = "conv-1";

const setup = (conversationId: string | null) => {
  const onSubmit = vi.fn();
  const { result } = renderHook(() =>
    useFeedbackInterceptor(conversationId, onSubmit),
  );
  return { intercept: result.current, onSubmit };
};

describe("useFeedbackInterceptor", () => {
  let openSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    openSpy = vi.spyOn(window, "open").mockImplementation(() => null);
  });

  afterEach(() => {
    openSpy.mockRestore();
  });

  it("passes a non-feedback message straight through to onSubmit", () => {
    const { intercept, onSubmit } = setup(CONV);
    intercept("hello there");
    expect(onSubmit).toHaveBeenCalledWith("hello there");
    expect(openSpy).not.toHaveBeenCalled();
  });

  it("opens the anonymous feedback form in a new tab", () => {
    const { intercept, onSubmit } = setup(CONV);
    intercept("/feedback");
    expect(openSpy).toHaveBeenCalledWith(FEEDBACK_FORM_URL, "_blank");
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("passes through (no window.open) when there is no conversation", () => {
    const { intercept, onSubmit } = setup(null);
    intercept("/feedback");
    expect(onSubmit).toHaveBeenCalledWith("/feedback");
    expect(openSpy).not.toHaveBeenCalled();
  });
});
