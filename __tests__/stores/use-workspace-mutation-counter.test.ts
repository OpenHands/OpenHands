import { afterEach, describe, expect, it, vi } from "vitest";

async function loadFreshCounterStore() {
  vi.resetModules();
  return (await import("#/stores/use-workspace-mutation-counter"))
    .useWorkspaceMutationCounter;
}

describe("workspace mutation counter", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("starts each page load at a cache-buster no earlier load used", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-01-01T00:00:00Z"));
    const firstLoad = await loadFreshCounterStore();
    vi.setSystemTime(new Date("2026-01-01T00:00:05Z"));
    const secondLoad = await loadFreshCounterStore();

    // A reload must not rebuild the `?v=` URL the previous load used, or the
    // rich preview can be answered from the browser HTTP cache (#17921).
    expect(secondLoad.getState().count).not.toBe(firstLoad.getState().count);
  });

  it("ticks forward on every bump", async () => {
    const useWorkspaceMutationCounter = await loadFreshCounterStore();
    const initial = useWorkspaceMutationCounter.getState().count;

    useWorkspaceMutationCounter.getState().bump();

    expect(useWorkspaceMutationCounter.getState().count).toBe(initial + 1);
  });
});

describe("withWorkspaceCacheBuster", () => {
  it("appends ?v= to http(s) URLs", async () => {
    const { withWorkspaceCacheBuster } =
      await import("#/stores/use-workspace-mutation-counter");
    expect(withWorkspaceCacheBuster("https://agent.example/a.png", 7)).toBe(
      "https://agent.example/a.png?v=7",
    );
  });

  it("leaves data: and blob: URLs untouched — appending ?v= corrupts them", async () => {
    const { withWorkspaceCacheBuster } =
      await import("#/stores/use-workspace-mutation-counter");
    const dataUrl = "data:image/png;base64,iVBORw0KGgo=";
    const blobUrl = "blob:https://agent.example/abc";
    expect(withWorkspaceCacheBuster(dataUrl, 7)).toBe(dataUrl);
    expect(withWorkspaceCacheBuster(blobUrl, 7)).toBe(blobUrl);
  });

  it("returns null for a null input", async () => {
    const { withWorkspaceCacheBuster } =
      await import("#/stores/use-workspace-mutation-counter");
    expect(withWorkspaceCacheBuster(null, 7)).toBe(null);
  });
});
