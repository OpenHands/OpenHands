import { renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { useClampDragSelection } from "./use-clamp-drag-selection";

interface Fixture {
  feed: HTMLDivElement;
  message: HTMLParagraphElement;
  code: HTMLElement;
  headerText: HTMLSpanElement;
  chromeText: HTMLSpanElement;
}

/** Header / feed / composer, matching the chat pane's DOM order. */
function mountChatPane(): Fixture {
  const header = document.createElement("div");
  const headerText = document.createElement("span");
  headerText.textContent = "Conversation name";
  header.append(headerText);

  const feed = document.createElement("div");
  feed.setAttribute("data-testid", "chat-scroll-container");
  const message = document.createElement("p");
  message.textContent = "Local pagination message 100";
  feed.append(message);
  const code = document.createElement("pre");
  code.textContent = "const answer = 42;";
  feed.append(code);
  // jsdom reports a zero-size rect; give the feed real vertical bounds so the
  // pointer position maps to the top/bottom edge the way it does in the pane.
  feed.getBoundingClientRect = () =>
    ({
      top: 100,
      bottom: 400,
      left: 0,
      right: 300,
      width: 300,
      height: 300,
      x: 0,
      y: 100,
      toJSON: () => ({}),
    }) as DOMRect;

  const chrome = document.createElement("div");
  const chromeText = document.createElement("span");
  chromeText.textContent = "Disconnected";
  chrome.append(chromeText);

  document.body.append(header, feed, chrome);
  return { feed, message, code, headerText, chromeText };
}

/** Drag from `anchor` to `focus`, which is how a mouse drag builds a selection. */
function dragSelection(anchor: Node, focus: Node, focusOffset: number) {
  const selection = window.getSelection()!;
  selection.collapse(anchor, 0);
  selection.extend(focus, focusOffset);
  return selection;
}

function mouseDownOn(target: Element, clientY: number) {
  target.dispatchEvent(
    new MouseEvent("mousedown", { bubbles: true, button: 0, clientY }),
  );
}

function dragPointerTo(clientY: number) {
  document.dispatchEvent(
    new MouseEvent("mousemove", { bubbles: true, clientY }),
  );
  document.dispatchEvent(new Event("selectionchange"));
}

describe("useClampDragSelection", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
  });

  afterEach(() => {
    window.getSelection()?.removeAllRanges();
    document.body.innerHTML = "";
  });

  it("keeps the selection in the feed when the drag escapes past the bottom edge", () => {
    const { feed, message, chromeText } = mountChatPane();
    renderHook(() => useClampDragSelection({ current: feed }));

    mouseDownOn(feed, 10);
    const selection = dragSelection(
      message.firstChild!,
      chromeText.firstChild!,
      5,
    );
    expect(feed.contains(selection.focusNode)).toBe(false);

    dragPointerTo(900);

    expect(feed.contains(selection.focusNode)).toBe(true);
    expect(selection.toString()).toContain("Local pagination message 100");
    expect(selection.toString()).not.toContain("Disconnected");
  });

  it("keeps the selection in the feed when the drag escapes past the top edge", () => {
    const { feed, message, headerText } = mountChatPane();
    renderHook(() => useClampDragSelection({ current: feed }));

    mouseDownOn(feed, 400);
    const selection = dragSelection(
      message.firstChild!,
      headerText.firstChild!,
      5,
    );
    expect(feed.contains(selection.focusNode)).toBe(false);

    dragPointerTo(0);

    expect(feed.contains(selection.focusNode)).toBe(true);
    expect(selection.toString()).not.toContain("Conversation name");
  });

  it("keeps the browser anchor when the drag escapes past the top edge", () => {
    const { feed, message, headerText } = mountChatPane();
    renderHook(() => useClampDragSelection({ current: feed }));

    mouseDownOn(feed, 300);
    const selection = dragSelection(
      message.firstChild!,
      headerText.firstChild!,
      5,
    );

    dragPointerTo(0);

    // The browser anchor must stay on the original text, not be moved to the
    // feed element: native movement resumes from the anchor when the pointer
    // re-enters the feed, so a container anchor would restart at message 1.
    expect(selection.anchorNode).toBe(message.firstChild);
    expect(feed.contains(selection.focusNode)).toBe(true);
  });

  it("leaves a sideways escape untouched", () => {
    const { feed, message, chromeText } = mountChatPane();
    renderHook(() => useClampDragSelection({ current: feed }));

    mouseDownOn(feed, 250);
    const selection = dragSelection(
      message.firstChild!,
      chromeText.firstChild!,
      5,
    );

    // The pointer stays between the feed's vertical bounds, so this is not a
    // top/bottom escape and must not be clamped to the feed's first message.
    dragPointerTo(250);

    expect(feed.contains(selection.focusNode)).toBe(false);
  });

  it("keeps the selection stable across repeated clamp updates", () => {
    const { feed, message, chromeText } = mountChatPane();
    renderHook(() => useClampDragSelection({ current: feed }));

    mouseDownOn(feed, 10);
    const selection = dragSelection(
      message.firstChild!,
      chromeText.firstChild!,
      5,
    );

    dragPointerTo(900);
    const afterFirst = selection.toString();
    // The browser keeps emitting updates as the auto-scroll continues; a
    // second clamp must not collapse the range to the container boundary.
    dragPointerTo(900);
    dragPointerTo(900);

    expect(selection.isCollapsed).toBe(false);
    expect(selection.toString()).toBe(afterFirst);
    expect(feed.contains(selection.focusNode)).toBe(true);
  });

  it("does not clamp a selection when the drag started outside the feed", () => {
    const { feed, headerText, chromeText } = mountChatPane();
    renderHook(() => useClampDragSelection({ current: feed }));

    mouseDownOn(chromeText, 10);
    const selection = dragSelection(
      chromeText.firstChild!,
      headerText.firstChild!,
      5,
    );

    dragPointerTo(0);

    expect(feed.contains(selection.focusNode)).toBe(false);
  });

  it("leaves composer selection untouched when the drag starts and ends in the composer", () => {
    const { feed, chromeText } = mountChatPane();
    renderHook(() => useClampDragSelection({ current: feed }));

    // A drag that begins in the composer must keep the composer text
    // selectable, even while the pointer is dragged over the feed's edge.
    mouseDownOn(chromeText, 500);
    const selection = dragSelection(
      chromeText.firstChild!,
      chromeText.firstChild!,
      12,
    );
    expect(selection.toString()).toBe("Disconnected");

    dragPointerTo(900);
    dragPointerTo(0);

    expect(feed.contains(selection.focusNode)).toBe(false);
    expect(selection.isCollapsed).toBe(false);
    expect(selection.toString()).toBe("Disconnected");
  });

  it("leaves an in-feed code block selection untouched when the drag stays inside the feed", () => {
    const { feed, code } = mountChatPane();
    renderHook(() => useClampDragSelection({ current: feed }));

    // Both ends are inside the feed with the pointer between the feed's
    // vertical bounds, so a code block stays selectable and the range is not
    // rewritten to the container boundary.
    mouseDownOn(code, 250);
    const selection = dragSelection(code.firstChild!, code.firstChild!, 18);
    expect(selection.toString()).toBe("const answer = 42;");

    dragPointerTo(250);

    expect(feed.contains(selection.focusNode)).toBe(true);
    expect(selection.isCollapsed).toBe(false);
    expect(selection.toString()).toBe("const answer = 42;");
  });

  it("stops clamping once the drag ends", () => {
    const { feed, message, chromeText } = mountChatPane();
    renderHook(() => useClampDragSelection({ current: feed }));

    mouseDownOn(feed, 10);
    document.dispatchEvent(new MouseEvent("mouseup", { bubbles: true }));
    const selection = dragSelection(
      message.firstChild!,
      chromeText.firstChild!,
      5,
    );

    dragPointerTo(900);

    expect(feed.contains(selection.focusNode)).toBe(false);
  });

  it("clamps on the next animation frame even without a selectionchange event", async () => {
    const { feed, message, chromeText } = mountChatPane();
    renderHook(() => useClampDragSelection({ current: feed }));

    mouseDownOn(feed, 10);
    const selection = dragSelection(
      message.firstChild!,
      chromeText.firstChild!,
      5,
    );
    // Move the pointer past the bottom edge but emit no `selectionchange`: the
    // browser can repaint an escaped focus without firing one, so the clamp
    // must also run from the frame loop.
    document.dispatchEvent(
      new MouseEvent("mousemove", { bubbles: true, clientY: 900 }),
    );
    expect(feed.contains(selection.focusNode)).toBe(false);

    await new Promise((resolve) => requestAnimationFrame(resolve));

    expect(feed.contains(selection.focusNode)).toBe(true);
    expect(selection.toString()).not.toContain("Disconnected");
  });

  it("does not start a frame loop before the drag begins", async () => {
    const { feed, message, chromeText } = mountChatPane();
    renderHook(() => useClampDragSelection({ current: feed }));

    const selection = dragSelection(
      message.firstChild!,
      chromeText.firstChild!,
      5,
    );
    document.dispatchEvent(
      new MouseEvent("mousemove", { bubbles: true, clientY: 900 }),
    );

    await new Promise((resolve) => requestAnimationFrame(resolve));

    // No primary-button drag started inside the feed, so nothing is clamped.
    expect(feed.contains(selection.focusNode)).toBe(false);
  });
});
