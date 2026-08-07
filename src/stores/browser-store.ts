import { create } from "zustand";

export type BrowserPanelMode = "empty" | "live" | "screenshot";

interface BrowserState {
  // empty: nothing to show; live: iframe of `iframeSrc`; screenshot: agent browser tool image
  mode: BrowserPanelMode;
  // URL shown in the chrome bar (may track in-iframe navigations when same-origin).
  url: string;
  // Value bound to the iframe `src` attribute. Only changes on explicit
  // parent-driven navigations (address bar, open_url, store back/forward,
  // reload remount) — never on syncLiveUrl, so in-iframe history stays intact.
  iframeSrc: string;
  // Base64-encoded screenshot of the browser window, when the tool provides one.
  screenshotSrc: string;
  // Live-preview navigation stack (address-bar / open_url / synced iframe loads).
  history: string[];
  historyIndex: number;
  // Bumped to force-remount the live iframe when location.reload() is unavailable.
  reloadToken: number;
}

interface BrowserStore extends BrowserState {
  setUrl: (url: string) => void;
  setLiveUrl: (url: string) => void;
  /** Sync chrome-bar URL after an in-iframe navigation (same-origin loads). */
  syncLiveUrl: (url: string) => void;
  setScreenshotSrc: (screenshotSrc: string) => void;
  goBack: () => void;
  goForward: () => void;
  reload: () => void;
  reset: () => void;
}

const initialState: BrowserState = {
  mode: "empty",
  url: "",
  iframeSrc: "",
  screenshotSrc: "",
  history: [],
  historyIndex: -1,
  reloadToken: 0,
};

function pushLiveUrl(
  state: BrowserState,
  url: string,
  options: { updateIframeSrc: boolean },
): Partial<BrowserState> {
  const truncated = state.history.slice(0, state.historyIndex + 1);
  const history =
    truncated[truncated.length - 1] === url ? truncated : [...truncated, url];
  const historyIndex = history.length - 1;

  return {
    mode: "live",
    url,
    screenshotSrc: "",
    history,
    historyIndex,
    ...(options.updateIframeSrc ? { iframeSrc: url } : {}),
  };
}

export const useBrowserStore = create<BrowserStore>((set) => ({
  ...initialState,
  setUrl: (url: string) =>
    set((state) => {
      // Agent browser-tool navigation takes over from a live iframe so we
      // don't keep embedding the previous open_url target.
      if (state.mode === "live") {
        return {
          mode: "empty",
          url,
          iframeSrc: "",
          screenshotSrc: "",
          history: [],
          historyIndex: -1,
        };
      }
      return { url };
    }),
  setLiveUrl: (url: string) =>
    set((state) => pushLiveUrl(state, url, { updateIframeSrc: true })),
  syncLiveUrl: (url: string) =>
    set((state) => {
      if (state.mode !== "live" || state.url === url) {
        return state;
      }
      // Update the address bar / history only — do not touch iframeSrc.
      return pushLiveUrl(state, url, { updateIframeSrc: false });
    }),
  setScreenshotSrc: (screenshotSrc: string) =>
    set({
      mode: "screenshot",
      screenshotSrc,
      history: [],
      historyIndex: -1,
    }),
  goBack: () =>
    set((state) => {
      if (state.mode !== "live" || state.historyIndex <= 0) {
        return state;
      }
      const historyIndex = state.historyIndex - 1;
      const url = state.history[historyIndex];
      return {
        historyIndex,
        url,
        iframeSrc: url,
        screenshotSrc: "",
      };
    }),
  goForward: () =>
    set((state) => {
      if (
        state.mode !== "live" ||
        state.historyIndex < 0 ||
        state.historyIndex >= state.history.length - 1
      ) {
        return state;
      }
      const historyIndex = state.historyIndex + 1;
      const url = state.history[historyIndex];
      return {
        historyIndex,
        url,
        iframeSrc: url,
        screenshotSrc: "",
      };
    }),
  reload: () =>
    set((state) => {
      if (state.mode !== "live" || !state.iframeSrc) {
        return state;
      }
      return { reloadToken: state.reloadToken + 1 };
    }),
  reset: () => set(initialState),
}));
