import { act, renderHook } from "@testing-library/react";
import type { RefObject } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useChatSubmission } from "#/hooks/chat/use-chat-submission";
import { useConversationStore } from "#/stores/conversation-store";

interface SubmissionProps {
  chatInputRef: RefObject<HTMLDivElement | null>;
  fileInputRef: RefObject<HTMLInputElement | null>;
  smartResize: () => void;
  onSubmit: (message: string) => void;
  resetManualResize?: () => void;
}

const createChatInput = (message: string) => {
  const element = document.createElement("div");
  element.contentEditable = "true";
  element.textContent = message;
  Object.defineProperty(element, "innerText", {
    configurable: true,
    writable: true,
    value: message,
  });
  document.body.appendChild(element);

  return {
    element,
    ref: { current: element } as RefObject<HTMLDivElement | null>,
  };
};

const createFileInput = (fileName: string) => {
  const element = document.createElement("input");
  element.type = "file";
  Object.defineProperty(element, "value", {
    configurable: true,
    writable: true,
    value: `C:\\fakepath\\${fileName}`,
  });
  document.body.appendChild(element);

  return {
    element,
    ref: { current: element } as RefObject<HTMLInputElement | null>,
  };
};

const renderSubmission = (props: SubmissionProps) =>
  renderHook(
    (currentProps: SubmissionProps) =>
      useChatSubmission(
        currentProps.chatInputRef,
        currentProps.fileInputRef,
        currentProps.smartResize,
        currentProps.onSubmit,
        currentProps.resetManualResize,
      ),
    { initialProps: props },
  );

const setAttachments = ({
  images = [],
  files = [],
}: {
  images?: File[];
  files?: File[];
} = {}) => useConversationStore.setState({ images, files });

afterEach(() => {
  setAttachments();
  document.body.replaceChildren();
  vi.restoreAllMocks();
});

describe("chat submission behavior", () => {
  it("keeps a blank draft untouched when there are no attachments", () => {
    setAttachments();
    const chatInput = createChatInput("  \n  ");
    const fileInput = createFileInput("draft.txt");
    const onSubmit = vi.fn();
    const smartResize = vi.fn();
    const resetManualResize = vi.fn();
    const { result } = renderSubmission({
      chatInputRef: chatInput.ref,
      fileInputRef: fileInput.ref,
      smartResize,
      onSubmit,
      resetManualResize,
    });

    act(() => result.current.handleSubmit());

    expect(onSubmit).not.toHaveBeenCalled();
    expect(chatInput.element.textContent).toBe("  \n  ");
    expect(fileInput.element.value).toBe("C:\\fakepath\\draft.txt");
    expect(smartResize).not.toHaveBeenCalled();
    expect(resetManualResize).not.toHaveBeenCalled();
  });

  it("submits the exact draft and clears both inputs before resetting size", () => {
    setAttachments();
    const chatInput = createChatInput("  Keep my spacing  \n");
    const fileInput = createFileInput("notes.txt");
    const callOrder: string[] = [];
    const onSubmit = vi.fn((message: string) => {
      callOrder.push(`submit:${message}`);
    });
    const smartResize = vi.fn(() => callOrder.push("resize"));
    const resetManualResize = vi.fn(() => callOrder.push("reset"));
    const { result } = renderSubmission({
      chatInputRef: chatInput.ref,
      fileInputRef: fileInput.ref,
      smartResize,
      onSubmit,
      resetManualResize,
    });

    act(() => result.current.handleSubmit());

    expect(onSubmit).toHaveBeenCalledWith("  Keep my spacing  \n");
    expect(chatInput.element.textContent).toBe("");
    expect(fileInput.element.value).toBe("");
    expect(callOrder).toEqual([
      "submit:  Keep my spacing  \n",
      "resize",
      "reset",
    ]);
  });

  it("submits an image without text", () => {
    const image = new File(["image"], "diagram.png", { type: "image/png" });
    setAttachments({ images: [image] });
    const chatInput = createChatInput("");
    const onSubmit = vi.fn();
    const smartResize = vi.fn();
    const { result } = renderSubmission({
      chatInputRef: chatInput.ref,
      fileInputRef: { current: null },
      smartResize,
      onSubmit,
    });

    act(() => result.current.handleSubmit());

    expect(onSubmit).toHaveBeenCalledWith("");
    expect(smartResize).toHaveBeenCalledOnce();
  });

  it("submits a regular file when the chat input is not mounted", () => {
    const file = new File(["notes"], "notes.txt", { type: "text/plain" });
    setAttachments({ files: [file] });
    const onSubmit = vi.fn();
    const smartResize = vi.fn();
    const { result } = renderSubmission({
      chatInputRef: { current: null },
      fileInputRef: { current: null },
      smartResize,
      onSubmit,
    });

    act(() => result.current.handleSubmit());

    expect(onSubmit).toHaveBeenCalledWith("");
    expect(smartResize).toHaveBeenCalledOnce();
  });

  it("runs the stop callback when provided and remains safe without one", () => {
    const onStop = vi.fn();
    const { result } = renderSubmission({
      chatInputRef: { current: null },
      fileInputRef: { current: null },
      smartResize: vi.fn(),
      onSubmit: vi.fn(),
    });

    act(() => {
      result.current.handleStop(onStop);
      result.current.handleStop();
    });

    expect(onStop).toHaveBeenCalledOnce();
  });

  it("uses the latest inputs and callbacks after rerender", () => {
    setAttachments();
    const firstChatInput = createChatInput("first draft");
    const firstFileInput = createFileInput("first.txt");
    const firstSubmit = vi.fn();
    const firstResize = vi.fn();
    const firstReset = vi.fn();
    const hook = renderSubmission({
      chatInputRef: firstChatInput.ref,
      fileInputRef: firstFileInput.ref,
      smartResize: firstResize,
      onSubmit: firstSubmit,
      resetManualResize: firstReset,
    });
    const latestChatInput = createChatInput("latest draft");
    const latestFileInput = createFileInput("latest.txt");
    const latestSubmit = vi.fn();
    const latestResize = vi.fn();
    const latestReset = vi.fn();

    hook.rerender({
      chatInputRef: latestChatInput.ref,
      fileInputRef: latestFileInput.ref,
      smartResize: latestResize,
      onSubmit: latestSubmit,
      resetManualResize: latestReset,
    });
    act(() => hook.result.current.handleSubmit());

    expect(firstSubmit).not.toHaveBeenCalled();
    expect(firstResize).not.toHaveBeenCalled();
    expect(firstReset).not.toHaveBeenCalled();
    expect(firstChatInput.element.textContent).toBe("first draft");
    expect(firstFileInput.element.value).toBe("C:\\fakepath\\first.txt");
    expect(latestSubmit).toHaveBeenCalledWith("latest draft");
    expect(latestResize).toHaveBeenCalledOnce();
    expect(latestReset).toHaveBeenCalledOnce();
    expect(latestChatInput.element.textContent).toBe("");
    expect(latestFileInput.element.value).toBe("");
  });
});
