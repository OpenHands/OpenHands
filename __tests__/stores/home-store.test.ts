import { describe, expect, it, vi } from "vitest";
import type { GitRepository } from "#/types/git";

const STORAGE_KEY = "home-store";
const SCOPE = "backend-a/org-a";

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
  // @spec BM-002 — Recent repositories belong to their backend and organization
  it("keeps independently persisted recents for each scope", async () => {
    const useHomeStore = await loadFreshHomeStore();
    const first = repository("first");
    const second = repository("second");
    useHomeStore.getState().addRecentRepository(first, "backend-a/org-a");
    useHomeStore.getState().addRecentRepository(second, "backend-a/org-b");
    await useHomeStore.persist.rehydrate();

    expect(
      useHomeStore.getState().getRecentRepositories("backend-a/org-a"),
    ).toEqual([first]);
    expect(
      useHomeStore.getState().getRecentRepositories("backend-a/org-b"),
    ).toEqual([second]);
    expect(
      useHomeStore.getState().getRecentRepositories("backend-b/org-a"),
    ).toEqual([]);
  });

  it("starts with no recent repositories or selected provider", async () => {
    const useHomeStore = await loadFreshHomeStore();

    expect(useHomeStore.getState().getRecentRepositories(SCOPE)).toEqual([]);
    expect(useHomeStore.getState().getLastSelectedProvider()).toBeNull();
  });

  it("drops legacy unscoped recents while preserving the provider preference", async () => {
    const useHomeStore = await loadFreshHomeStore();
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        state: {
          recentRepositories: [repository("legacy")],
          lastSelectedProvider: "gitlab",
        },
        version: 0,
      }),
    );

    await useHomeStore.persist.rehydrate();

    expect(useHomeStore.getState().getRecentRepositories(SCOPE)).toEqual([]);
    expect(useHomeStore.getState().getLastSelectedProvider()).toBe("gitlab");
    expect(
      JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "{}").state,
    ).not.toHaveProperty("recentRepositories");
  });

  it("clears only the requested scope and distinguishes provider IDs", async () => {
    const useHomeStore = await loadFreshHomeStore();
    const github = repository("shared-id");
    const gitlab = repository("shared-id", { git_provider: "gitlab" });
    const store = useHomeStore.getState();
    store.addRecentRepository(github, SCOPE);
    store.addRecentRepository(gitlab, SCOPE);
    store.addRecentRepository(github, "other-scope");

    expect(store.getRecentRepositories(SCOPE)).toEqual([gitlab, github]);
    store.clearRecentRepositories("other-scope");

    expect(store.getRecentRepositories("other-scope")).toEqual([]);
    expect(store.getRecentRepositories(SCOPE)).toEqual([gitlab, github]);
  });

  it("adds a repository to the front and persists it", async () => {
    const useHomeStore = await loadFreshHomeStore();
    const first = repository("first");

    useHomeStore.getState().addRecentRepository(first, SCOPE);

    expect(useHomeStore.getState().getRecentRepositories(SCOPE)).toEqual([
      first,
    ]);
    expect(
      JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "{}"),
    ).toEqual({
      state: {
        lastSelectedProvider: null,
        recentRepositoriesByScope: { [SCOPE]: [first] },
      },
      version: 1,
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

    useHomeStore.getState().addRecentRepository(first, SCOPE);
    useHomeStore.getState().addRecentRepository(second, SCOPE);
    useHomeStore.getState().addRecentRepository(updatedFirst, SCOPE);

    expect(useHomeStore.getState().getRecentRepositories(SCOPE)).toEqual([
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
      useHomeStore.getState().addRecentRepository(item, SCOPE);
    }

    expect(useHomeStore.getState().getRecentRepositories(SCOPE)).toEqual([
      repositories[3],
      repositories[2],
      repositories[1],
    ]);
  });

  it("clears all recent repositories", async () => {
    const useHomeStore = await loadFreshHomeStore();
    useHomeStore.getState().addRecentRepository(repository("first"), SCOPE);

    useHomeStore.getState().clearRecentRepositories(SCOPE);

    expect(useHomeStore.getState().getRecentRepositories(SCOPE)).toEqual([]);
    expect(
      JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "{}"),
    ).toMatchObject({ state: { recentRepositoriesByScope: { [SCOPE]: [] } } });
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
