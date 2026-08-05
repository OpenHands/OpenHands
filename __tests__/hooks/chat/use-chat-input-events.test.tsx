import { act, renderHook } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { useChatInputEvents } from "#/hooks/chat/use-chat-input-events";
import type React from "react";

const DESKTOP_UA =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36";
const MOBILE_UA =
  "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1";

function setUserAgent(ua: string) {
  Object.defineProperty(window.navigator, "userAgent", {
    value: ua,
    configurable: true,
    writable: true,
  });
}

function makeEnterEvent(overrides: Partial<React.KeyboardEvent> = {}) {
  return {
    key: "Enter",
    shiftKey: false,
    preventDefault: vi.fn(),
    nativeEvent: { isComposing: false },
    ...overrides,
  } as unknown as React.KeyboardEvent;
}

describe("useChatInputEvents — Enter on an empty input", () => {
  const smartResize = vi.fn();
  const increaseHeightForEmptyContent = vi.fn();
  const checkIsContentEmpty = vi.fn();
  const clearEmptyContentHandler = vi.fn();
  const onFocus = vi.fn();
  const onBlur = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    checkIsContentEmpty.mockReturnValue(true);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  function renderEvents() {
    const ref = { current: document.createElement("div") };
    const { result } = renderHook(() =>
      useChatInputEvents(
        ref,
        smartResize,
        increaseHeightForEmptyContent,
        checkIsContentEmpty,
        clearEmptyContentHandler,
        onFocus,
        onBlur,
      ),
    );
    return result;
  }

  it("grows the empty composer on desktop so Shift+Enter-style expansion still works", () => {
    setUserAgent(DESKTOP_UA);
    const result = renderEvents();
    const event = makeEnterEvent();

    act(() => {
      result.current.handleKeyDown(event, false, vi.fn());
    });

    expect(event.preventDefault).toHaveBeenCalled();
    expect(increaseHeightForEmptyContent).toHaveBeenCalledTimes(1);
  });

  it("does not grow the empty composer on phones/tablets (Enter is a newline key)", () => {
    setUserAgent(MOBILE_UA);
    const result = renderEvents();
    const event = makeEnterEvent();

    act(() => {
      result.current.handleKeyDown(event, false, vi.fn());
    });

    expect(event.preventDefault).toHaveBeenCalled();
    expect(increaseHeightForEmptyContent).not.toHaveBeenCalled();
  });
});
