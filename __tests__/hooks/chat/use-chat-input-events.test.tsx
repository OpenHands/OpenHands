import { act, renderHook } from "@testing-library/react";
import type { RefObject } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useChatInputEvents } from "#/hooks/chat/use-chat-input-events";
import { useGripResize } from "#/hooks/chat/use-grip-resize";
import { useConversationStore } from "#/stores/conversation-store";

const createElementRef = (text = "message") => {
  const element = document.createElement("div");
  element.contentEditable = "true";
  element.textContent = text;
  document.body.appendChild(element);
  return {
    element,
    ref: { current: element } as RefObject<HTMLDivElement | null>,
  };
};

const createKeyboardEvent = ({
  key = "Enter",
  composing = false,
  shiftKey = false,
}: {
  key?: string;
  composing?: boolean;
  shiftKey?: boolean;
} = {}) =>
  ({
    key,
    shiftKey,
    nativeEvent: { isComposing: composing },
    preventDefault: vi.fn(),
  }) as unknown as React.KeyboardEvent;

const createClipboardEvent = ({
  files = [],
  text = "",
}: {
  files?: File[];
  text?: string;
}) =>
  ({
    preventDefault: vi.fn(),
    clipboardData: {
      files,
      items: [],
      getData: vi.fn((format: string) => (format === "text/plain" ? text : "")),
    },
  }) as unknown as React.ClipboardEvent;

const renderInputEvents = ({
  inputRef = { current: null },
  smartResize = vi.fn(),
  increaseHeight = vi.fn(),
  isEmpty = vi.fn(() => false),
  onFocus,
  onBlur,
}: {
  inputRef?: RefObject<HTMLDivElement | null>;
  smartResize?: () => void;
  increaseHeight?: () => void;
  isEmpty?: () => boolean;
  onFocus?: () => void;
  onBlur?: () => void;
} = {}) =>
  renderHook(() =>
    useChatInputEvents(
      inputRef,
      smartResize,
      increaseHeight,
      isEmpty,
      onFocus,
      onBlur,
    ),
  );

afterEach(() => {
  document.body.replaceChildren();
  Reflect.deleteProperty(document, "execCommand");
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("chat input DOM events", () => {
  it("does not notify conversation subscribers while typing at the same suggestion visibility", () => {
    vi.useFakeTimers();
    const input = createElementRef("Draft");
    let contentHeight = 20;
    Object.defineProperty(input.element, "scrollHeight", {
      configurable: true,
      get: () => contentHeight,
    });
    useConversationStore.setState({ shouldHideSuggestions: false });
    const { result } = renderHook(() => {
      const { smartResize, increaseHeightForEmptyContent } = useGripResize(
        input.ref,
        null,
      );
      return useChatInputEvents(
        input.ref,
        smartResize,
        increaseHeightForEmptyContent,
        () => false,
      );
    });
    act(() => vi.runOnlyPendingTimers());
    const notifyConversation = vi.fn();
    const unsubscribe = useConversationStore.subscribe(notifyConversation);

    try {
      for (const text of ["Draft a", "Draft a response", "Draft a response."]) {
        input.element.textContent = text;
        act(() => result.current.handleInput());
        act(() => vi.runOnlyPendingTimers());
      }

      expect(notifyConversation).not.toHaveBeenCalled();

      // Expanding the composer must still hide the empty-state suggestions.
      contentHeight = 180;
      act(() => result.current.handleInput());
      act(() => vi.runOnlyPendingTimers());
      expect(useConversationStore.getState().shouldHideSuggestions).toBe(true);

      // Further typing at that height must leave unrelated consumers alone.
      notifyConversation.mockClear();
      input.element.textContent += " More detail.";
      act(() => result.current.handleInput());
      act(() => vi.runOnlyPendingTimers());
      expect(notifyConversation).not.toHaveBeenCalled();

      contentHeight = 20;
      act(() => result.current.handleInput());
      act(() => vi.runOnlyPendingTimers());
      expect(useConversationStore.getState().shouldHideSuggestions).toBe(false);
    } finally {
      unsubscribe();
    }
  });

  it("resizes and clears whitespace while remaining safe before mount", () => {
    const mounted = createElementRef("   ");
    const mountedResize = vi.fn();
    const mountedHook = renderInputEvents({
      inputRef: mounted.ref,
      smartResize: mountedResize,
    });
    const unmountedResize = vi.fn();
    const unmountedHook = renderInputEvents({
      smartResize: unmountedResize,
    });

    act(() => {
      mountedHook.result.current.handleInput();
      unmountedHook.result.current.handleInput();
    });

    expect(mountedResize).toHaveBeenCalledOnce();
    expect(unmountedResize).toHaveBeenCalledOnce();
    expect(mounted.element.textContent).toBe("");
    expect(mounted.element.innerHTML).toBe("");
  });

  it("dispatches pasted files to the attachment system", () => {
    const { result } = renderInputEvents();
    const image = new File(["image"], "diagram.png", { type: "image/png" });
    const paste = createClipboardEvent({ files: [image] });
    const pastedFiles = vi.fn();
    document.addEventListener("pasteFiles", pastedFiles);

    try {
      act(() => result.current.handlePaste(paste));

      expect(paste.preventDefault).toHaveBeenCalledOnce();
      expect(pastedFiles).toHaveBeenCalledOnce();
      expect(
        (pastedFiles.mock.calls[0][0] as CustomEvent<{ files: File[] }>).detail
          .files,
      ).toEqual([image]);
    } finally {
      document.removeEventListener("pasteFiles", pastedFiles);
    }
  });

  it("inserts pasted text and resizes on the next task", () => {
    vi.useFakeTimers();
    const smartResize = vi.fn();
    const { result } = renderInputEvents({ smartResize });
    const execCommand = vi.fn();
    Object.defineProperty(document, "execCommand", {
      configurable: true,
      value: execCommand,
    });
    const textPaste = createClipboardEvent({ text: "plain text" });

    act(() => result.current.handlePaste(textPaste));
    expect(execCommand).toHaveBeenCalledWith("insertText", false, "plain text");
    expect(smartResize).not.toHaveBeenCalled();

    act(() => vi.runOnlyPendingTimers());
    expect(smartResize).toHaveBeenCalledOnce();
  });

  it("does not resize or insert an empty non-file paste", () => {
    vi.useFakeTimers();
    const smartResize = vi.fn();
    const { result } = renderInputEvents({ smartResize });
    const execCommand = vi.fn();
    Object.defineProperty(document, "execCommand", {
      configurable: true,
      value: execCommand,
    });
    const paste = createClipboardEvent({ text: "" });

    act(() => result.current.handlePaste(paste));
    act(() => vi.runOnlyPendingTimers());

    expect(paste.preventDefault).toHaveBeenCalledOnce();
    expect(execCommand).not.toHaveBeenCalled();
    expect(smartResize).not.toHaveBeenCalled();
  });

  it("clears whitespace on blur and invokes optional focus callbacks", () => {
    const input = createElementRef("  ");
    const onFocus = vi.fn();
    const onBlur = vi.fn();
    const { result } = renderInputEvents({
      inputRef: input.ref,
      onFocus,
      onBlur,
    });
    const withoutOptions = renderInputEvents();

    expect(() =>
      act(() => {
        result.current.handleFocus();
        result.current.handleBlur();
        withoutOptions.result.current.handleFocus();
        withoutOptions.result.current.handleBlur();
      }),
    ).not.toThrow();
    expect(onFocus).toHaveBeenCalledOnce();
    expect(onBlur).toHaveBeenCalledOnce();
    expect(input.element.textContent).toBe("");
  });
});

describe("chat input keyboard events", () => {
  it("ignores non-Enter keys and active IME composition", () => {
    const isEmpty = vi.fn(() => false);
    const { result } = renderInputEvents({ isEmpty });
    const letter = createKeyboardEvent({ key: "a" });
    const composingEnter = createKeyboardEvent({ composing: true });
    const submit = vi.fn();

    act(() => {
      result.current.handleKeyDown(letter, false, submit);
      result.current.handleKeyDown(composingEnter, false, submit);
    });

    expect(letter.preventDefault).not.toHaveBeenCalled();
    expect(composingEnter.preventDefault).not.toHaveBeenCalled();
    expect(isEmpty).not.toHaveBeenCalled();
    expect(submit).not.toHaveBeenCalled();
  });

  it("increases the input height instead of submitting empty content", () => {
    const increaseHeight = vi.fn();
    const { result } = renderInputEvents({
      increaseHeight,
      isEmpty: () => true,
    });
    const event = createKeyboardEvent();
    const submit = vi.fn();

    act(() => result.current.handleKeyDown(event, false, submit));

    expect(event.preventDefault).toHaveBeenCalledOnce();
    expect(increaseHeight).toHaveBeenCalledOnce();
    expect(submit).not.toHaveBeenCalled();
  });

  it("submits non-empty content on desktop Enter", () => {
    vi.spyOn(window.navigator, "userAgent", "get").mockReturnValue(
      "Mozilla/5.0 (X11; Linux x86_64)",
    );
    const { result } = renderInputEvents();
    const event = createKeyboardEvent();
    const submit = vi.fn();

    act(() => result.current.handleKeyDown(event, false, submit));

    expect(event.preventDefault).toHaveBeenCalledOnce();
    expect(submit).toHaveBeenCalledOnce();
  });

  it.each([
    {
      label: "mobile Enter",
      userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X)",
      shiftKey: false,
      disabled: false,
    },
    {
      label: "Shift+Enter",
      userAgent: "Mozilla/5.0 (X11; Linux x86_64)",
      shiftKey: true,
      disabled: false,
    },
    {
      label: "disabled Enter",
      userAgent: "Mozilla/5.0 (X11; Linux x86_64)",
      shiftKey: false,
      disabled: true,
    },
  ])("leaves $label unchanged", ({ userAgent, shiftKey, disabled }) => {
    vi.spyOn(window.navigator, "userAgent", "get").mockReturnValue(userAgent);
    const { result } = renderInputEvents();
    const event = createKeyboardEvent({ shiftKey });
    const submit = vi.fn();

    act(() => result.current.handleKeyDown(event, disabled, submit));

    expect(event.preventDefault).not.toHaveBeenCalled();
    expect(submit).not.toHaveBeenCalled();
  });
});

describe("chat input callback freshness", () => {
  it("uses the latest input ref and resize callback after rerender", () => {
    vi.useFakeTimers();
    Object.defineProperty(document, "execCommand", {
      configurable: true,
      value: vi.fn(),
    });
    const first = createElementRef("first");
    const latest = createElementRef("   ");
    const firstResize = vi.fn();
    const latestResize = vi.fn();
    const increaseHeight = vi.fn();
    const isEmpty = vi.fn(() => false);
    const { result, rerender } = renderHook(
      ({ inputRef, resize }) =>
        useChatInputEvents(inputRef, resize, increaseHeight, isEmpty),
      {
        initialProps: { inputRef: first.ref, resize: firstResize },
      },
    );

    rerender({ inputRef: latest.ref, resize: latestResize });
    act(() => result.current.handleInput());
    act(() =>
      result.current.handlePaste(createClipboardEvent({ text: "latest" })),
    );
    act(() => vi.runOnlyPendingTimers());

    expect(latest.element.textContent).toBe("");
    expect(firstResize).not.toHaveBeenCalled();
    expect(latestResize).toHaveBeenCalledTimes(2);
  });

  it("uses the latest empty-content callbacks after rerender", () => {
    const inputRef: RefObject<HTMLDivElement | null> = { current: null };
    const smartResize = vi.fn();
    const firstCheck = vi.fn(() => false);
    const latestCheck = vi.fn(() => true);
    const firstIncrease = vi.fn();
    const latestIncrease = vi.fn();
    const { result, rerender } = renderHook(
      ({ check, increase }) =>
        useChatInputEvents(inputRef, smartResize, increase, check),
      {
        initialProps: { check: firstCheck, increase: firstIncrease },
      },
    );
    const event = createKeyboardEvent();
    const submit = vi.fn();

    rerender({ check: latestCheck, increase: latestIncrease });
    act(() => result.current.handleKeyDown(event, false, submit));

    expect(firstCheck).not.toHaveBeenCalled();
    expect(firstIncrease).not.toHaveBeenCalled();
    expect(latestCheck).toHaveBeenCalledOnce();
    expect(latestIncrease).toHaveBeenCalledOnce();
    expect(submit).not.toHaveBeenCalled();
  });

  it("uses the latest focus, blur, and input ref after rerender", () => {
    const first = createElementRef("first");
    const latest = createElementRef("   ");
    const firstFocus = vi.fn();
    const latestFocus = vi.fn();
    const firstBlur = vi.fn();
    const latestBlur = vi.fn();
    const smartResize = vi.fn();
    const increaseHeight = vi.fn();
    const isEmpty = vi.fn(() => false);
    const { result, rerender } = renderHook(
      ({ inputRef, focus, blur }) =>
        useChatInputEvents(
          inputRef,
          smartResize,
          increaseHeight,
          isEmpty,
          focus,
          blur,
        ),
      {
        initialProps: {
          inputRef: first.ref,
          focus: firstFocus,
          blur: firstBlur,
        },
      },
    );

    rerender({ inputRef: latest.ref, focus: latestFocus, blur: latestBlur });
    act(() => {
      result.current.handleFocus();
      result.current.handleBlur();
    });

    expect(firstFocus).not.toHaveBeenCalled();
    expect(firstBlur).not.toHaveBeenCalled();
    expect(latestFocus).toHaveBeenCalledOnce();
    expect(latestBlur).toHaveBeenCalledOnce();
    expect(latest.element.textContent).toBe("");
  });
});
