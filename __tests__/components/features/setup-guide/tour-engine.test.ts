import { waitFor } from "@testing-library/react";
import type { Config } from "driver.js";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { notifySuperAdminSetupStep } from "#/components/features/setup-guide/super-admin-setup-step-event";
import {
  isGuidedTourActive,
  startGuidedTour,
  stopGuidedTour,
} from "#/components/features/setup-guide/tour/tour-engine";
import type {
  GuidedTour,
  GuidedTourStop,
} from "#/components/features/setup-guide/tour/tour-types";

const { drivers } = vi.hoisted(() => ({
  drivers: [] as {
    destroy: ReturnType<typeof vi.fn>;
    moveTo: ReturnType<typeof vi.fn>;
    config: Config;
  }[],
}));

// Like driver.js, destroying a tour runs its onDestroyed hook.
vi.mock("driver.js", () => ({
  driver: (config: Config) => {
    const instance = {
      drive: vi.fn(),
      refresh: vi.fn(),
      moveTo: vi.fn(),
      moveNext: vi.fn(),
      movePrevious: vi.fn(),
      destroy: vi.fn(() => config.onDestroyed?.(undefined, {}, {} as never)),
      config,
    };
    drivers.push(instance);
    return instance;
  },
}));

const LABELS = { next: "Next", previous: "Back", done: "Done" };
const NAV = { navigate: vi.fn(), getPath: () => "/" };

function stop(overrides: Partial<GuidedTourStop>): GuidedTourStop {
  return {
    id: "stop",
    anchor: "#anchor",
    title: "Title",
    body: "Body",
    ...overrides,
  };
}

function tour(...stops: GuidedTourStop[]): GuidedTour {
  return { id: "tour", stops };
}

const wait = (ms: number) =>
  new Promise((resolve) => {
    setTimeout(resolve, ms);
  });

describe("startGuidedTour", () => {
  beforeEach(() => {
    document.body.innerHTML = `
      <section id="anchor">
        <button id="card" type="button">Template</button>
        <input id="search" />
      </section>
      <form id="form"></form>`;
    Element.prototype.scrollIntoView = vi.fn();
    drivers.length = 0;
  });

  afterEach(() => {
    stopGuidedTour();
    vi.useRealTimers();
    document.body.innerHTML = "";
  });

  const pickThenDialog = () =>
    tour(
      stop({ id: "pick", advanceOnClick: "#card" }),
      stop({ id: "dialog", anchor: "#dialog" }),
    );

  it("ends the tour when the step its last stop waits for succeeds", async () => {
    // Arrange
    await startGuidedTour(
      tour(stop({ waitForComplete: "add-integration" })),
      NAV,
      LABELS,
    );
    await wait(60); // the stop starts waiting once it is shown

    // Act
    notifySuperAdminSetupStep("add-integration");

    // Assert
    await waitFor(() => expect(isGuidedTourActive()).toBe(false));
    expect(drivers[0].destroy).toHaveBeenCalledTimes(1);
  });

  it("does not let a replaced tour act on the tour that replaced it", async () => {
    // Arrange: the first tour's stop is done, so it is about to end itself.
    await startGuidedTour(
      tour(stop({ waitForComplete: "first-automation" })),
      NAV,
      LABELS,
    );
    await wait(60);
    notifySuperAdminSetupStep("first-automation");

    // Act: the next step's tour starts before that happens.
    await startGuidedTour(tour(stop({ id: "next" })), NAV, LABELS);
    await wait(150);

    // Assert
    expect(isGuidedTourActive()).toBe(true);
    expect(drivers[0].destroy).toHaveBeenCalledTimes(1);
    expect(drivers[1].destroy).not.toHaveBeenCalled();
  });

  it("moves on when a matching element in the anchor is clicked, not on other clicks", async () => {
    // Arrange
    await startGuidedTour(
      tour(
        stop({ id: "pick", advanceOnClick: "#card" }),
        stop({ id: "form", anchor: "#form" }),
      ),
      NAV,
      LABELS,
    );
    await wait(60);

    // Act
    document.getElementById("search")!.click();
    await wait(350);
    const afterOtherClick = drivers[0].moveTo.mock.calls.length;
    document.getElementById("card")!.click();

    // Assert
    expect(afterOtherClick).toBe(0);
    await waitFor(() => expect(drivers[0].moveTo).toHaveBeenCalledWith(1));
  });

  it("hides while a click opens something else first, then shows the next stop once its element appears", async () => {
    // Arrange
    await startGuidedTour(pickThenDialog(), NAV, LABELS);
    await wait(60);

    // Act: the click opens a choice first, so the next stop's dialog is late.
    document.getElementById("card")!.click();
    await wait(1_400);

    // Assert: the tour is hidden and not marked active, so the setup guide
    // can show; nothing has moved yet.
    expect(document.body).toHaveClass("oh-setup-tour-waiting");
    expect(isGuidedTourActive()).toBe(false);
    expect(drivers[0].moveTo).not.toHaveBeenCalled();

    // Act
    document.body.insertAdjacentHTML("beforeend", '<div id="dialog"></div>');

    // Assert
    await waitFor(() => expect(drivers[0].moveTo).toHaveBeenCalledWith(1));
    expect(document.body).not.toHaveClass("oh-setup-tour-waiting");
    expect(isGuidedTourActive()).toBe(true);
  });

  it("ends the tour when the next stop's element never appears", async () => {
    // Arrange
    vi.useFakeTimers();
    await startGuidedTour(pickThenDialog(), NAV, LABELS);
    await vi.advanceTimersByTimeAsync(60);

    // Act: the admin went another way, so the dialog never opens.
    document.getElementById("card")!.click();
    await vi.advanceTimersByTimeAsync(61_000);

    // Assert: never shown away from its element.
    expect(drivers[0].moveTo).not.toHaveBeenCalled();
    expect(drivers[0].destroy).toHaveBeenCalledTimes(1);
    expect(document.body).not.toHaveClass("oh-setup-tour-waiting");
    expect(isGuidedTourActive()).toBe(false);
  });

  it("lets Tab move through the page, not the hidden tour, while it waits", async () => {
    // Arrange: driver.js keeps Tab in its stop with a listener on window.
    const tourTabListener = vi.fn();
    window.addEventListener("keydown", tourTabListener);
    const pressTab = () =>
      document.body.dispatchEvent(
        new KeyboardEvent("keydown", { key: "Tab", bubbles: true }),
      );
    await startGuidedTour(pickThenDialog(), NAV, LABELS);
    await wait(60);

    try {
      // Act
      document.getElementById("card")!.click();
      await wait(350);
      pressTab();
      stopGuidedTour();
      pressTab();

      // Assert
      expect(tourTabListener).toHaveBeenCalledTimes(1);
    } finally {
      window.removeEventListener("keydown", tourTabListener);
    }
  });

  it("does not start when the first stop's element is missing", async () => {
    // Arrange
    vi.useFakeTimers();

    // Act
    const started = startGuidedTour(
      tour(stop({ anchor: "#missing" })),
      NAV,
      LABELS,
    );
    await vi.advanceTimersByTimeAsync(6_000);
    await started;

    // Assert: driver.js would float the popover over the page.
    expect(drivers).toHaveLength(0);
    expect(isGuidedTourActive()).toBe(false);
  });

  it("hides the popover footer only on a stop that asks for it", async () => {
    // Arrange
    const renderPopover = () => {
      const popover = {
        nextButton: document.createElement("button"),
        footer: document.createElement("div"),
      };
      drivers[0].config.onPopoverRender?.(popover as never, {} as never);
      return popover;
    };
    await startGuidedTour(
      tour(
        stop({ id: "pick", advanceOnClick: "#card", hideFooter: true }),
        stop({ id: "form", anchor: "#form" }),
      ),
      NAV,
      LABELS,
    );
    await wait(60);

    // Act
    const first = renderPopover();
    document.getElementById("card")!.click();
    await waitFor(() => expect(drivers[0].moveTo).toHaveBeenCalledWith(1));
    const second = renderPopover();

    // Assert
    expect(first.footer.style.display).toBe("none");
    expect(second.footer.style.display).toBe("");
  });

  it("keeps the page usable while an interactive stop is shown", async () => {
    // Act
    await startGuidedTour(
      tour(stop({ anchor: "#form", interactive: true })),
      NAV,
      LABELS,
    );

    // Assert
    expect(document.body).toHaveClass("oh-setup-tour-interactive");
    stopGuidedTour();
    expect(document.body).not.toHaveClass("oh-setup-tour-interactive");
  });

  it("opens the first stop's route when the admin is elsewhere", async () => {
    // Arrange
    const navigate = vi.fn();

    // Act
    await startGuidedTour(
      tour(stop({ route: "/mcp" })),
      { navigate, getPath: () => "/conversations" },
      LABELS,
    );

    // Assert
    expect(navigate).toHaveBeenCalledWith("/mcp");
  });
});
