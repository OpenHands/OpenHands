import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

import { browserTab } from "#/utils/browser-tab";

// These tests exercise the browser-tab notification flasher behavior.
// Specifically we verify that when the document title changes externally
// while a notification is active, the flasher updates its internal
// baseline so it restores/toggles to the new title instead of an old one.

describe("browserTab notifications", () => {
  const MESSAGE = "Agent ready";
  const INITIAL = "Conversation 123 | OpenHands";
  const RENAMED = "My renamed title | OpenHands";

  beforeEach(() => {
    vi.useFakeTimers();
    // reset title for each test
    document.title = INITIAL;
  });

  afterEach(() => {
    browserTab.stopNotification();
    vi.runOnlyPendingTimers();
    vi.useRealTimers();
  });

  it("flashes the browser tab title between the original title and the notification message", () => {
    browserTab.startNotification(MESSAGE);

    // First tick: should switch to the notification message
    vi.advanceTimersByTime(1000);
    expect(document.title).toBe(MESSAGE);

    // Next tick: should switch back to original
    vi.advanceTimersByTime(1000);
    expect(document.title).toBe(INITIAL);

    // Next tick: should switch to message again
    vi.advanceTimersByTime(1000);
    expect(document.title).toBe(MESSAGE);
  });

  it("stops flashing and restores original title when stopNotification is called", () => {
    browserTab.startNotification(MESSAGE);

    vi.advanceTimersByTime(1000);
    expect(document.title).toBe(MESSAGE);

    browserTab.stopNotification();
    expect(document.title).toBe(INITIAL);
  });

  it("updates baseline when title changes during an active notification and restores to the new title", () => {
    // Start flashing
    browserTab.startNotification(MESSAGE);

    // Tick once: should switch to the message
    vi.advanceTimersByTime(1000);
    expect(document.title).toBe(MESSAGE);

    // Simulate an external rename while flashing (e.g., user edits title)
    document.title = RENAMED;

    // Next tick: flasher observes the external change and updates baseline
    vi.advanceTimersByTime(1000);
    // On this tick, we toggle back to the message
    expect(document.title).toBe(MESSAGE);

    // Next tick should toggle to the updated baseline (renamed title)
    vi.advanceTimersByTime(1000);
    expect(document.title).toBe(RENAMED);

    // Stop flashing: title should remain the updated baseline
    browserTab.stopNotification();
    expect(document.title).toBe(RENAMED);
  });
});
