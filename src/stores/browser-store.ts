import { create } from "zustand";

interface BrowserState {
  // URL of the last page the agent navigated to in the browser panel.
  url: string;
  // Base64-encoded screenshot of the browser window, when the tool provides one.
  screenshotSrc: string;
}

interface BrowserStore extends BrowserState {
  setUrl: (url: string) => void;
  setScreenshotSrc: (screenshotSrc: string) => void;
  reset: () => void;
}

const initialState: BrowserState = {
  url: "",
  screenshotSrc: "",
};

export const useBrowserStore = create<BrowserStore>((set) => ({
  ...initialState,
  setUrl: (url: string) => set({ url }),
  setScreenshotSrc: (screenshotSrc: string) => set({ screenshotSrc }),
  reset: () => set(initialState),
}));
