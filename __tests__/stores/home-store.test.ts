import { describe, expect, it, vi } from "vitest";
import type { GitRepository } from "#/types/git";

const STORAGE_KEY = "home-store";

const repository = (
  id: string,
  overrides: Partial<GitRepository> = {},
): GitRepository => ({
  id,
  full_name: `owner/${id}`,
  git_provider: "github",
  is_public: true,
  ...overrides,
});

async function loadFreshHomeStore() {
  window.localStorage.clear();
  vi.resetModules();
  return (await import("#/stores/home-store")).useHomeStore;
}

describe("home store", () => {
  it("starts with no recent repositories or selected provider", async () => {
    const useHomeStore = await loadFreshHomeStore();

    expect(useHomeStore.getState().getRecentRepositories()).toEqual([]);
    expect(useHomeStore.getState().getLastSelectedProvider()).toBeNull();
  });

  it("adds a repository to the front and persists it", async () => {
    const useHomeStore = await loadFreshHomeStore();
    const first = repository("first");

    useHomeStore.getState().addRecentRepository(first);

    expect(useHomeStore.getState().getRecentRepositories()).toEqual([first]);
    expect(
      JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "{}"),
    ).toEqual({
      state: {
        lastSelectedProvider: null,
        recentRepositories: [first],
      },
      version: 0,
    });
  });

  it("moves an existing repository to the front without duplicating it", async () => {
    const useHomeStore = await loadFreshHomeStore();
    const first = repository("first");
    const second = repository("second");
    const updatedFirst = repository("first", {
      full_name: "new-owner/first",
      stargazers_count: 7,
    });

    useHomeStore.getState().addRecentRepository(first);
    useHomeStore.getState().addRecentRepository(second);
    useHomeStore.getState().addRecentRepository(updatedFirst);

    expect(useHomeStore.getState().getRecentRepositories()).toEqual([
      updatedFirst,
      second,
    ]);
  });

  it("keeps only the three most recently selected repositories", async () => {
    const useHomeStore = await loadFreshHomeStore();
    const repositories = ["first", "second", "third", "fourth"].map((id) =>
      repository(id),
    );

    for (const item of repositories) {
      useHomeStore.getState().addRecentRepository(item);
    }

    expect(useHomeStore.getState().getRecentRepositories()).toEqual([
      repositories[3],
      repositories[2],
      repositories[1],
    ]);
  });

  it("clears all recent repositories", async () => {
    const useHomeStore = await loadFreshHomeStore();
    useHomeStore.getState().addRecentRepository(repository("first"));

    useHomeStore.getState().clearRecentRepositories();

    expect(useHomeStore.getState().getRecentRepositories()).toEqual([]);
    expect(
      JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "{}"),
    ).toMatchObject({ state: { recentRepositories: [] } });
  });

  it("sets, reads, persists, and clears the last selected provider", async () => {
    const useHomeStore = await loadFreshHomeStore();
    useHomeStore.getState().setLastSelectedProvider("gitlab");

    expect(useHomeStore.getState().getLastSelectedProvider()).toBe("gitlab");
    expect(
      JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "{}"),
    ).toMatchObject({ state: { lastSelectedProvider: "gitlab" } });

    useHomeStore.getState().setLastSelectedProvider(null);

    expect(useHomeStore.getState().getLastSelectedProvider()).toBeNull();
  });
});
