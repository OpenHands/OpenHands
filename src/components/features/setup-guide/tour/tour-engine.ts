import { driver, type Driver } from "driver.js";
import "driver.js/dist/driver.css";
import { SUPER_ADMIN_SETUP_STEP_EVENT } from "../super-admin-setup-step-event";
import type {
  GuidedTour,
  GuidedTourLabels,
  GuidedTourSide,
  GuidedTourStop,
} from "./tour-types";
import "./tour-theme.css";

/**
 * Spotlight tour for the enterprise setup guide's steps in Canvas, ported from
 * the enterprise app's tour engine so both apps' tours look and behave alike.
 */

/** On `<body>` while the current stop keeps the whole page usable. */
const INTERACTIVE_STOP_CLASS = "oh-setup-tour-interactive";

/** On `<body>` while the tour is hidden, looking for the next stop's element. */
const WAITING_CLASS = "oh-setup-tour-waiting";

/**
 * How long a stop reached by a click may take to appear. The click can open
 * something else first, such as a choice of setup or an install dialog, and
 * the admin may take a while there.
 */
const NEXT_STOP_TIMEOUT_MS = 60_000;

export interface GuidedTourNavigation {
  navigate: (to: string) => void;
  /** The current route, relative to the app's base path. */
  getPath: () => string;
}

let activeDriver: Driver | null = null;
const resolvedAnchors = new Map<string, Element>();
const tourActiveListeners = new Set<() => void>();
let tourActive = false;
let setupStepListener: ((event: Event) => void) | null = null;
// Bumped whenever a tour is stopped or replaced. driver.js keeps its state at
// module level, so a delayed callback from an older tour would act on (and
// destroy) the current one; callbacks check their generation first.
let tourGeneration = 0;

function setTourActive(next: boolean): void {
  if (tourActive === next) {
    return;
  }
  tourActive = next;
  tourActiveListeners.forEach((listener) => listener());
}

export function isGuidedTourActive(): boolean {
  return tourActive;
}

export function subscribeGuidedTourActive(listener: () => void): () => void {
  tourActiveListeners.add(listener);
  return () => {
    tourActiveListeners.delete(listener);
  };
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function setInteractive(stop: GuidedTourStop | undefined): void {
  document.body.classList.toggle(
    INTERACTIVE_STOP_CLASS,
    Boolean(stop?.interactive),
  );
}

// driver.js keeps Tab inside the stop it last showed, with a listener on
// `window`. While the tour is hidden, Tab should move through what the click
// opened instead, so it is stopped on `document` first.
function keepTabOutOfHiddenTour(event: KeyboardEvent): void {
  if (event.key === "Tab") {
    event.stopPropagation();
  }
}

function setWaiting(waiting: boolean): void {
  document.body.classList.toggle(WAITING_CLASS, waiting);
  if (waiting) {
    document.addEventListener("keydown", keepTabOutOfHiddenTour);
  } else {
    document.removeEventListener("keydown", keepTabOutOfHiddenTour);
  }
}

function queryFirst(selector: string): Element | null {
  const parts = selector.split(",").map((part) => part.trim());
  for (const part of parts) {
    const el = document.querySelector(part);
    if (el) {
      return el;
    }
  }
  return null;
}

async function resolveAnchor(
  selector: string,
  timeoutMs: number,
): Promise<Element | null> {
  // A stop reached by a click (a new route, a dialog) can take a moment.
  for (let attempt = 0; attempt * 100 < timeoutMs; attempt += 1) {
    const el = queryFirst(selector);
    if (el) {
      el.scrollIntoView({ block: "nearest", inline: "nearest" });
      return el;
    }
    await wait(100);
  }
  return null;
}

function sideFor(el: Element, requested?: GuidedTourSide): GuidedTourSide {
  const rect = el.getBoundingClientRect();
  const spaceRight = window.innerWidth - rect.right;
  const spaceBottom = window.innerHeight - rect.bottom;
  if (requested === "right" && spaceRight > 340) {
    return "right";
  }
  if (requested === "left" && rect.left > 340) {
    return "left";
  }
  if (spaceBottom > 220) {
    return "bottom";
  }
  if (rect.top > 220) {
    return "top";
  }
  return requested ?? "bottom";
}

async function prepareStop(
  stop: GuidedTourStop,
  nav: GuidedTourNavigation,
  timeoutMs = 5_000,
): Promise<Element | null> {
  if (stop.route && nav.getPath() !== stop.route) {
    nav.navigate(stop.route);
    // The route's page needs a beat before its anchors exist.
    await wait(400);
  }
  const el = await resolveAnchor(stop.anchor, timeoutMs);
  if (el) {
    resolvedAnchors.set(stop.id, el);
  } else {
    resolvedAnchors.delete(stop.id);
  }
  return el;
}

function clearSetupStepListener(): void {
  if (setupStepListener) {
    window.removeEventListener(SUPER_ADMIN_SETUP_STEP_EVENT, setupStepListener);
    setupStepListener = null;
  }
}

export function stopGuidedTour(): void {
  tourGeneration += 1;
  clearSetupStepListener();
  setInteractive(undefined);
  setWaiting(false);
  activeDriver?.destroy();
  activeDriver = null;
  resolvedAnchors.clear();
  setTourActive(false);
}

export async function startGuidedTour(
  tour: GuidedTour,
  nav: GuidedTourNavigation,
  labels: GuidedTourLabels,
): Promise<void> {
  stopGuidedTour();
  const generation = tourGeneration;
  const isCurrent = () => generation === tourGeneration;

  const { stops } = tour;
  if (stops.length === 0) {
    return;
  }
  // A stop is only ever shown on its element: without one, driver.js would
  // float the popover over whatever is on the page.
  const firstAnchor = await prepareStop(stops[0], nav);
  if (!isCurrent() || !firstAnchor) {
    return;
  }

  let index = 0;
  let anchorClickHandler: ((event: Event) => void) | null = null;
  const isWaiting = (stop: GuidedTourStop | undefined) =>
    Boolean(stop?.waitForComplete || stop?.advanceOnClick);

  const tourCtl: {
    clearAnchorClickHandler: () => void;
    advanceFrom: (instanceApi: Driver, nextIndex: number) => Promise<void>;
    bindWaitForComplete: (instanceApi: Driver) => void;
    bindInteractiveStop: (instanceApi: Driver) => void;
  } = {
    clearAnchorClickHandler: () => {
      if (!anchorClickHandler) {
        return;
      }
      for (const el of resolvedAnchors.values()) {
        el.removeEventListener("click", anchorClickHandler);
      }
      anchorClickHandler = null;
    },
    advanceFrom: async (instanceApi, nextIndex) => {
      if (!isCurrent()) {
        return;
      }
      const nextStop = stops[nextIndex];
      if (!nextStop) {
        instanceApi.destroy();
        return;
      }
      clearSetupStepListener();
      tourCtl.clearAnchorClickHandler();
      // Hide the tour while the next stop's element is looked for, so it
      // covers nothing the click opened first and the page stays usable. A
      // longer wait also lets the setup guide show again meanwhile.
      setWaiting(true);
      const markIdle = window.setTimeout(() => {
        if (isCurrent()) setTourActive(false);
      }, 1_000);
      const anchor = await prepareStop(nextStop, nav, NEXT_STOP_TIMEOUT_MS);
      window.clearTimeout(markIdle);
      if (!isCurrent()) {
        return;
      }
      setWaiting(false);
      if (!anchor) {
        // The flow went elsewhere, e.g. the admin chose the built-in
        // integration or a template that starts a conversation.
        instanceApi.destroy();
        return;
      }
      setTourActive(true);
      index = nextIndex;
      setInteractive(nextStop);
      instanceApi.moveTo(nextIndex);
      window.setTimeout(() => {
        if (!isCurrent()) {
          return;
        }
        instanceApi.refresh();
        tourCtl.bindInteractiveStop(instanceApi);
      }, 30);
    },
    bindWaitForComplete: (instanceApi) => {
      clearSetupStepListener();
      const waitIndex = stops.findIndex((stop) => stop.waitForComplete);
      if (waitIndex < 0) {
        return;
      }
      const { waitForComplete } = stops[waitIndex];
      setupStepListener = (event: Event) => {
        const { detail } = event as CustomEvent<{ id?: string }>;
        if (detail?.id !== waitForComplete || index > waitIndex) {
          return;
        }
        clearSetupStepListener();
        // Let the completing UI (a dialog) unmount before moving on.
        wait(80)
          .then(() => tourCtl.advanceFrom(instanceApi, waitIndex + 1))
          .catch(() => undefined);
      };
      window.addEventListener(SUPER_ADMIN_SETUP_STEP_EVENT, setupStepListener);
    },
    bindInteractiveStop: (instanceApi) => {
      tourCtl.clearAnchorClickHandler();
      tourCtl.bindWaitForComplete(instanceApi);
      const current = stops[index];
      const anchor = current && resolvedAnchors.get(current.id);
      if (!anchor || !current.advanceOnClick) {
        return;
      }
      anchorClickHandler = (event: Event) => {
        const { target } = event;
        if (
          current.advanceOnClick &&
          !(target instanceof Element && target.closest(current.advanceOnClick))
        ) {
          return;
        }
        tourCtl.clearAnchorClickHandler();
        wait(280)
          .then(() => tourCtl.advanceFrom(instanceApi, index + 1))
          .catch(() => undefined);
      };
      anchor.addEventListener("click", anchorClickHandler);
    },
  };

  const instance = driver({
    showProgress: stops.length > 1,
    progressText: "{{current}}/{{total}}",
    nextBtnText: labels.next,
    prevBtnText: labels.previous,
    doneBtnText: labels.done,
    allowClose: true,
    // Arrow keys would move the tour while the admin types in a form.
    allowKeyboardControl: false,
    // Allow clicking the highlighted control (a template card, a button).
    disableActiveInteraction: false,
    overlayColor: "#000",
    overlayOpacity: 0.6,
    stagePadding: 8,
    popoverOffset: 16,
    smoothScroll: true,
    popoverClass: "oh-setup-tour-popover",
    steps: stops.map((stop) => ({
      element: () => resolvedAnchors.get(stop.id) as Element,
      popover: {
        title: stop.title,
        description: stop.body,
        side: stop.side ?? "bottom",
        align: "start",
      },
    })),
    onHighlightStarted: (el, step) => {
      if (el && step.popover) {
        // eslint-disable-next-line no-param-reassign -- driver.js reads it back
        step.popover.side = sideFor(
          el,
          step.popover.side as GuidedTourSide | undefined,
        );
      }
    },
    onPopoverRender: (popover) => {
      // Completion and clicks are event-driven on these stops.
      // eslint-disable-next-line no-param-reassign -- driver.js owns this node
      popover.nextButton.style.display = isWaiting(stops[index]) ? "none" : "";
      // A per-step `showProgress: false` falls back to the tour-wide value in
      // driver.js, so the footer is hidden here instead.
      if (stops[index]?.hideFooter) {
        // eslint-disable-next-line no-param-reassign -- driver.js owns this node
        popover.footer.style.display = "none";
      }
    },
    onNextClick: async (_el, _step, { driver: instanceApi }) => {
      const current = stops[index];
      if (isWaiting(current)) {
        return;
      }
      tourCtl.clearAnchorClickHandler();
      await tourCtl.advanceFrom(instanceApi, index + 1);
    },
    onPrevClick: async (_el, _step, { driver: instanceApi }) => {
      const prevStop = stops[index - 1];
      if (!prevStop) {
        return;
      }
      clearSetupStepListener();
      tourCtl.clearAnchorClickHandler();
      const anchor = await prepareStop(prevStop, nav);
      if (!isCurrent()) {
        return;
      }
      if (!anchor) {
        instanceApi.destroy();
        return;
      }
      index -= 1;
      setInteractive(prevStop);
      instanceApi.movePrevious();
      window.setTimeout(() => {
        if (!isCurrent()) {
          return;
        }
        instanceApi.refresh();
        tourCtl.bindInteractiveStop(instanceApi);
      }, 30);
    },
    onCloseClick: (_el, _step, { driver: instanceApi }) => {
      instanceApi.destroy();
    },
    onDestroyed: () => {
      tourCtl.clearAnchorClickHandler();
      if (!isCurrent()) {
        return;
      }
      clearSetupStepListener();
      setInteractive(undefined);
      setWaiting(false);
      activeDriver = null;
      resolvedAnchors.clear();
      setTourActive(false);
    },
  });

  activeDriver = instance;
  setTourActive(true);
  setInteractive(stops[0]);
  instance.drive();
  window.setTimeout(() => {
    if (!isCurrent()) {
      return;
    }
    instance.refresh();
    tourCtl.bindInteractiveStop(instance);
  }, 40);
}
