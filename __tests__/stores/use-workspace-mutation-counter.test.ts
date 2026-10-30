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
