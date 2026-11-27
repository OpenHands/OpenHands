import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ChangeEvent, DragEvent as ReactDragEvent } from "react";
import { useFileHandling } from "#/hooks/chat/use-file-handling";

const createFile = (name: string) =>
  new File([`${name} contents`], name, { type: "text/plain" });

const createDragEvent = ({
  files = [],
  relatedTarget = null,
}: {
  files?: File[];
  relatedTarget?: EventTarget | null;
} = {}) =>
  ({
    preventDefault: vi.fn(),
    dataTransfer: { files },
    relatedTarget,
  }) as unknown as ReactDragEvent<HTMLDivElement>;

afterEach(() => {
  document.body.replaceChildren();
  vi.restoreAllMocks();
});

describe("file handling", () => {
  it("forwards pasted files with paste metadata and ignores empty paste events", () => {
    const addEventListener = vi.spyOn(document, "addEventListener");
    const onFilesPaste = vi.fn();
    renderHook(() => useFileHandling(onFilesPaste));
    const pastedFile = createFile("pasted.txt");

    act(() => {
      document.dispatchEvent(
        new CustomEvent("pasteFiles", {
          detail: { files: [pastedFile] },
        }),
      );
      document.dispatchEvent(
        new CustomEvent("pasteFiles", { detail: { files: [] } }),
      );
    });

    const pasteListener = addEventListener.mock.calls.find(
      ([eventName]) => eventName === "pasteFiles",
    )?.[1] as EventListener;
    expect(() =>
      act(() => {
        pasteListener(
          new CustomEvent("pasteFiles", { detail: { files: undefined } }),
        );
      }),
    ).not.toThrow();

    expect(onFilesPaste).toHaveBeenCalledOnce();
    expect(onFilesPaste).toHaveBeenCalledWith([pastedFile], {
      fromPaste: true,
    });
  });

  it("removes its paste listener on unmount", () => {
    const removeEventListener = vi.spyOn(document, "removeEventListener");
    const onFilesPaste = vi.fn();
    const { unmount } = renderHook(() => useFileHandling(onFilesPaste));

    unmount();
    document.dispatchEvent(
      new CustomEvent("pasteFiles", {
        detail: { files: [createFile("after-unmount.txt")] },
      }),
    );

    expect(removeEventListener).toHaveBeenCalledWith(
      "pasteFiles",
      expect.any(Function),
    );
    expect(onFilesPaste).not.toHaveBeenCalled();
  });

  it("accepts paste events when no upload callback is configured", () => {
    renderHook(() => useFileHandling());

    expect(() => {
      act(() => {
        document.dispatchEvent(
          new CustomEvent("pasteFiles", {
            detail: { files: [createFile("unhandled.txt")] },
          }),
        );
      });
    }).not.toThrow();
  });

  it("opens the file chooser only when uploads are enabled and the input is mounted", () => {
    const { result } = renderHook(() => useFileHandling());
    const input = document.createElement("input");
    const click = vi.spyOn(input, "click");

    act(() => result.current.handleFileIconClick(false));
    expect(click).not.toHaveBeenCalled();

    result.current.fileInputRef.current = input;
    act(() => result.current.handleFileIconClick(true));
    expect(click).not.toHaveBeenCalled();

    act(() => result.current.handleFileIconClick(false));
    expect(click).toHaveBeenCalledOnce();
  });

  it("forwards selected input files and ignores a cleared input", () => {
    const onFilesPaste = vi.fn();
    const { result } = renderHook(() => useFileHandling(onFilesPaste));
    const selectedFile = createFile("selected.txt");

    act(() => {
      result.current.handleFileInputChange({
        target: { files: [selectedFile] },
      } as unknown as ChangeEvent<HTMLInputElement>);
    });
    expect(onFilesPaste).toHaveBeenCalledWith([selectedFile], undefined);

    act(() => {
      result.current.handleFileInputChange({
        target: { files: null },
      } as unknown as ChangeEvent<HTMLInputElement>);
    });
    expect(onFilesPaste).toHaveBeenCalledOnce();
  });

  it("shows drag feedback only for enabled uploads", () => {
    const { result } = renderHook(() => useFileHandling());
    const disabledEvent = createDragEvent();

    act(() => result.current.handleDragOver(disabledEvent, true));
    expect(disabledEvent.preventDefault).not.toHaveBeenCalled();
    expect(result.current.isDragOver).toBe(false);

    const enabledEvent = createDragEvent();
    act(() => result.current.handleDragOver(enabledEvent, false));
    expect(enabledEvent.preventDefault).toHaveBeenCalledOnce();
    expect(result.current.isDragOver).toBe(true);
  });

  it("keeps drag feedback while disabled or moving within the chat container", () => {
    const { result } = renderHook(() => useFileHandling());
    const container = document.createElement("div");
    const child = document.createElement("span");
    container.appendChild(child);
    result.current.chatContainerRef.current = container;

    act(() => result.current.handleDragOver(createDragEvent(), false));

    const disabledEvent = createDragEvent();
    act(() => result.current.handleDragLeave(disabledEvent, true));
    expect(disabledEvent.preventDefault).not.toHaveBeenCalled();
    expect(result.current.isDragOver).toBe(true);

    const internalEvent = createDragEvent({ relatedTarget: child });
    act(() => result.current.handleDragLeave(internalEvent, false));
    expect(internalEvent.preventDefault).not.toHaveBeenCalled();
    expect(result.current.isDragOver).toBe(true);
  });

  it("clears drag feedback when the pointer leaves the container or no container is mounted", () => {
    const { result } = renderHook(() => useFileHandling());
    const container = document.createElement("div");
    const outside = document.createElement("div");
    result.current.chatContainerRef.current = container;

    act(() => result.current.handleDragOver(createDragEvent(), false));
    const outsideEvent = createDragEvent({ relatedTarget: outside });
    act(() => result.current.handleDragLeave(outsideEvent, false));
    expect(outsideEvent.preventDefault).toHaveBeenCalledOnce();
    expect(result.current.isDragOver).toBe(false);

    act(() => result.current.handleDragOver(createDragEvent(), false));
    result.current.chatContainerRef.current = null;
    const unmountedEvent = createDragEvent({ relatedTarget: outside });
    act(() => result.current.handleDragLeave(unmountedEvent, false));
    expect(unmountedEvent.preventDefault).toHaveBeenCalledOnce();
    expect(result.current.isDragOver).toBe(false);
  });

  it("ignores disabled drops without changing active drag feedback", () => {
    const onFilesPaste = vi.fn();
    const { result } = renderHook(() => useFileHandling(onFilesPaste));
    act(() => result.current.handleDragOver(createDragEvent(), false));
    const dropEvent = createDragEvent({ files: [createFile("ignored.txt")] });

    act(() => result.current.handleDrop(dropEvent, true));

    expect(dropEvent.preventDefault).not.toHaveBeenCalled();
    expect(onFilesPaste).not.toHaveBeenCalled();
    expect(result.current.isDragOver).toBe(true);
  });

  it("forwards enabled drops and always clears drag feedback", () => {
    const onFilesPaste = vi.fn();
    const { result } = renderHook(() => useFileHandling(onFilesPaste));
    const droppedFile = createFile("dropped.txt");
    act(() => result.current.handleDragOver(createDragEvent(), false));
    const dropEvent = createDragEvent({ files: [droppedFile] });

    act(() => result.current.handleDrop(dropEvent, false));

    expect(dropEvent.preventDefault).toHaveBeenCalledOnce();
    expect(onFilesPaste).toHaveBeenCalledWith([droppedFile], undefined);
    expect(result.current.isDragOver).toBe(false);

    act(() => result.current.handleDragOver(createDragEvent(), false));
    const emptyDropEvent = createDragEvent();
    act(() => result.current.handleDrop(emptyDropEvent, false));
    expect(emptyDropEvent.preventDefault).toHaveBeenCalledOnce();
    expect(onFilesPaste).toHaveBeenCalledOnce();
    expect(result.current.isDragOver).toBe(false);
  });

  it("uses the latest upload callback for paste, input, and drop after rerender", () => {
    const firstCallback = vi.fn();
    const latestCallback = vi.fn();
    const { result, rerender } = renderHook(
      ({ callback }) => useFileHandling(callback),
      { initialProps: { callback: firstCallback } },
    );
    const file = createFile("latest.txt");

    rerender({ callback: latestCallback });
    act(() => {
      document.dispatchEvent(
        new CustomEvent("pasteFiles", { detail: { files: [file] } }),
      );
      result.current.handleFileInputChange({
        target: { files: [file] },
      } as unknown as ChangeEvent<HTMLInputElement>);
      result.current.handleDrop(createDragEvent({ files: [file] }), false);
    });

    expect(firstCallback).not.toHaveBeenCalled();
    expect(latestCallback).toHaveBeenCalledTimes(3);
    expect(latestCallback).toHaveBeenNthCalledWith(1, [file], {
      fromPaste: true,
    });
    expect(latestCallback).toHaveBeenNthCalledWith(2, [file], undefined);
    expect(latestCallback).toHaveBeenNthCalledWith(3, [file], undefined);
  });
});
