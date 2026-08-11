import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useChatInputEvents } from "#/hooks/chat/use-chat-input-events";

// Avoid the actual mobile-device heuristic from leaking into this test
vi.mock("#/utils/utils", async (importOriginal) => {
  const mod = await importOriginal<typeof import("#/utils/utils")>();
  return { ...mod, isMobileDevice: () => false };
});

const noop = () => {};

const buildHarness = (overrides: { hasAttachments?: boolean } = {}) => {
  const chatInputRef = {
    current: null,
  } as React.RefObject<HTMLDivElement | null>;
  const checkIsContentEmpty = vi.fn(() => true);
  const handleSubmit = vi.fn();
  const increaseHeightForEmptyContent = vi.fn();
  const { result } = renderHook(() =>
    useChatInputEvents(
      chatInputRef,
      noop,
      increaseHeightForEmptyContent,
      checkIsContentEmpty,
      noop,
      undefined,
      undefined,
      overrides.hasAttachments ?? false,
    ),
  );
  return {
    result,
    checkIsContentEmpty,
    handleSubmit,
    increaseHeightForEmptyContent,
  };
};

const fireEnter = (
  handleKeyDown: ReturnType<typeof useChatInputEvents>["handleKeyDown"],
  handleSubmit: () => void,
  isComposing: boolean,
) => {
  let prevented = false;
  const event = {
    key: "Enter",
    shiftKey: false,
    nativeEvent: { isComposing },
    preventDefault: () => {
      prevented = true;
    },
  } as unknown as React.KeyboardEvent;
  act(() => {
    handleKeyDown(event, false, handleSubmit);
  });
  return { prevented };
};

describe("useChatInputEvents", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("grows the input on Enter when the text is empty and no attachments are queued", () => {
    const {
      result,
      checkIsContentEmpty,
      handleSubmit,
      increaseHeightForEmptyContent,
    } = buildHarness();
    const { prevented } = fireEnter(
      result.current.handleKeyDown,
      handleSubmit,
      false,
    );
    expect(checkIsContentEmpty).toHaveBeenCalled();
    expect(handleSubmit).not.toHaveBeenCalled();
    expect(increaseHeightForEmptyContent).toHaveBeenCalled();
    expect(prevented).toBe(true);
  });

  it("submits attachment-only messages on Enter", () => {
    const {
      result,
      checkIsContentEmpty,
      handleSubmit,
      increaseHeightForEmptyContent,
    } = buildHarness({ hasAttachments: true });
    const { prevented } = fireEnter(
      result.current.handleKeyDown,
      handleSubmit,
      false,
    );
    expect(checkIsContentEmpty).toHaveBeenCalled();
    expect(increaseHeightForEmptyContent).not.toHaveBeenCalled();
    expect(handleSubmit).toHaveBeenCalledTimes(1);
    expect(prevented).toBe(true);
  });

  it("ignores Enter while an IME composition is in progress", () => {
    const { result, handleSubmit, increaseHeightForEmptyContent } =
      buildHarness({ hasAttachments: true });
    const { prevented } = fireEnter(
      result.current.handleKeyDown,
      handleSubmit,
      true,
    );
    expect(handleSubmit).not.toHaveBeenCalled();
    expect(increaseHeightForEmptyContent).not.toHaveBeenCalled();
    expect(prevented).toBe(false);
  });
});
