import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { useSidebarStore } from "#/stores/sidebar-store";
import { useToggleSidebarShortcut } from "./use-toggle-sidebar-shortcut";

const STORAGE_KEY = "openhands-sidebar";

function pressKey(init: KeyboardEventInit, target: EventTarget = window) {
  const event = new KeyboardEvent("keydown", {
    bubbles: true,
    cancelable: true,
    ...init,
  });
  target.dispatchEvent(event);
  return event;
}

function collapsed() {
  return useSidebarStore.getState().collapsed;
}

describe("useToggleSidebarShortcut", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useSidebarStore.setState({ collapsed: false });
  });

  it("toggles the sidebar in both directions on Ctrl+Shift+S", () => {
    renderHook(() => useToggleSidebarShortcut());

    pressKey({ key: "S", ctrlKey: true, shiftKey: true });
    expect(collapsed()).toBe(true);

    pressKey({ key: "S", ctrlKey: true, shiftKey: true });
    expect(collapsed()).toBe(false);
  });

  it("accepts Cmd+Shift+S on apple platforms", () => {
    renderHook(() => useToggleSidebarShortcut());

    pressKey({ key: "S", metaKey: true, shiftKey: true });

    expect(collapsed()).toBe(true);
  });

  it("works while the chat composer has focus", () => {
    renderHook(() => useToggleSidebarShortcut());
    const composer = document.createElement("textarea");
    document.body.append(composer);
    composer.focus();

    // Bubbles up from the focused field, so the window listener still sees it.
    pressKey({ key: "S", ctrlKey: true, shiftKey: true }, composer);

    expect(collapsed()).toBe(true);
    composer.remove();
  });

  it("persists the new state the way the click toggle does", () => {
    renderHook(() => useToggleSidebarShortcut());

    pressKey({ key: "S", ctrlKey: true, shiftKey: true });

    const stored = JSON.parse(
      window.localStorage.getItem(STORAGE_KEY) ?? "{}",
    ) as { state?: { collapsed?: boolean } };
    expect(stored.state?.collapsed).toBe(true);
  });

  it("prevents the browser default for the handled combination", () => {
    renderHook(() => useToggleSidebarShortcut());

    const event = pressKey({ key: "S", ctrlKey: true, shiftKey: true });

    expect(event.defaultPrevented).toBe(true);
  });

  it("toggles once while the combination is held down", () => {
    renderHook(() => useToggleSidebarShortcut());

    pressKey({ key: "S", ctrlKey: true, shiftKey: true });
    pressKey({ key: "S", ctrlKey: true, shiftKey: true, repeat: true });
    pressKey({ key: "S", ctrlKey: true, shiftKey: true, repeat: true });

    expect(collapsed()).toBe(true);
  });

  it.each([
    ["a bare Shift+S, as typed in the composer", { key: "S", shiftKey: true }],
    ["Ctrl+S without Shift", { key: "s", ctrlKey: true }],
    ["Ctrl+Shift+A", { key: "a", ctrlKey: true, shiftKey: true }],
    [
      "Ctrl+Alt+Shift+S",
      { key: "S", ctrlKey: true, altKey: true, shiftKey: true },
    ],
  ])("ignores %s", (_label, init) => {
    renderHook(() => useToggleSidebarShortcut());

    pressKey(init);

    expect(collapsed()).toBe(false);
  });

  it("stops listening once unmounted", () => {
    const { unmount } = renderHook(() => useToggleSidebarShortcut());

    unmount();
    pressKey({ key: "S", ctrlKey: true, shiftKey: true });

    expect(collapsed()).toBe(false);
  });
});
