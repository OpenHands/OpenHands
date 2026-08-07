import { beforeEach, describe, expect, it } from "vitest";

import { useBrowserStore } from "#/stores/browser-store";

describe("useBrowserStore", () => {
  beforeEach(() => {
    useBrowserStore.getState().reset();
  });

  it("setLiveUrl enters live mode and clears any previous screenshot", () => {
    useBrowserStore.setState({
      mode: "screenshot",
      url: "https://old.example",
      screenshotSrc: "data:image/png;base64,abc",
    });

    useBrowserStore.getState().setLiveUrl("http://localhost:3000");

    expect(useBrowserStore.getState()).toMatchObject({
      mode: "live",
      url: "http://localhost:3000",
      iframeSrc: "http://localhost:3000",
      screenshotSrc: "",
      history: ["http://localhost:3000"],
      historyIndex: 0,
    });
  });

  it("syncLiveUrl updates the chrome bar without changing iframeSrc", () => {
    useBrowserStore.getState().setLiveUrl("http://localhost:3000");
    useBrowserStore.getState().syncLiveUrl("http://localhost:3000/about");

    expect(useBrowserStore.getState()).toMatchObject({
      mode: "live",
      url: "http://localhost:3000/about",
      iframeSrc: "http://localhost:3000",
      history: ["http://localhost:3000", "http://localhost:3000/about"],
      historyIndex: 1,
    });
  });

  it("setScreenshotSrc enters screenshot mode without clearing the url", () => {
    useBrowserStore.getState().setLiveUrl("http://localhost:3000");
    useBrowserStore.getState().setScreenshotSrc("data:image/png;base64,xyz");

    expect(useBrowserStore.getState()).toMatchObject({
      mode: "screenshot",
      url: "http://localhost:3000",
      screenshotSrc: "data:image/png;base64,xyz",
      history: [],
      historyIndex: -1,
    });
  });

  it("setUrl while live drops the iframe until a screenshot arrives", () => {
    useBrowserStore.getState().setLiveUrl("http://localhost:3000");
    useBrowserStore.getState().setUrl("https://agent-browsed.example");

    expect(useBrowserStore.getState()).toMatchObject({
      mode: "empty",
      url: "https://agent-browsed.example",
      iframeSrc: "",
      screenshotSrc: "",
      history: [],
      historyIndex: -1,
    });
  });

  it("tracks live history for back and forward navigation", () => {
    const store = useBrowserStore.getState();
    store.setLiveUrl("http://localhost:3000");
    store.setLiveUrl("http://localhost:3000/a");
    store.setLiveUrl("http://localhost:3000/b");

    useBrowserStore.getState().goBack();
    expect(useBrowserStore.getState()).toMatchObject({
      url: "http://localhost:3000/a",
      iframeSrc: "http://localhost:3000/a",
      historyIndex: 1,
    });

    useBrowserStore.getState().goBack();
    expect(useBrowserStore.getState()).toMatchObject({
      url: "http://localhost:3000",
      iframeSrc: "http://localhost:3000",
      historyIndex: 0,
    });

    useBrowserStore.getState().goForward();
    expect(useBrowserStore.getState()).toMatchObject({
      url: "http://localhost:3000/a",
      iframeSrc: "http://localhost:3000/a",
      historyIndex: 1,
    });
  });

  it("reload bumps reloadToken only in live mode", () => {
    useBrowserStore.getState().reload();
    expect(useBrowserStore.getState().reloadToken).toBe(0);

    useBrowserStore.getState().setLiveUrl("http://localhost:3000");
    useBrowserStore.getState().reload();
    expect(useBrowserStore.getState().reloadToken).toBe(1);
  });

  it("reset returns to the empty initial state", () => {
    useBrowserStore.getState().setLiveUrl("http://localhost:3000");
    useBrowserStore.getState().reset();

    expect(useBrowserStore.getState()).toMatchObject({
      mode: "empty",
      url: "",
      iframeSrc: "",
      screenshotSrc: "",
      history: [],
      historyIndex: -1,
      reloadToken: 0,
    });
  });
});
