import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import { GitRepository } from "#/types/git";
import { Provider } from "#/types/settings";

interface HomeState {
  recentRepositoriesByScope: Record<string, GitRepository[]>;
  lastSelectedProvider: Provider | null;
}

interface HomeActions {
  addRecentRepository: (repository: GitRepository, scope: string) => void;
  clearRecentRepositories: (scope: string) => void;
  getRecentRepositories: (scope: string) => GitRepository[];
  setLastSelectedProvider: (provider: Provider | null) => void;
  getLastSelectedProvider: () => Provider | null;
}

type HomeStore = HomeState & HomeActions;

const EMPTY_RECENT_REPOSITORIES: GitRepository[] = [];

const initialState: HomeState = {
  recentRepositoriesByScope: {},
  lastSelectedProvider: null,
};

export const useHomeStore = create<HomeStore>()(
  persist(
    (set, get) => ({
      ...initialState,

      // @spec BM-002 — Persist recents only within the scope that selected them
      addRecentRepository: (repository: GitRepository, scope: string) =>
        set((state) => {
          // Remove the repository if it already exists to avoid duplicates
          const filteredRepos = (
            state.recentRepositoriesByScope[scope] ?? []
          ).filter(
            (repo) =>
              repo.id !== repository.id ||
              repo.git_provider !== repository.git_provider,
          );

          // Add the new repository to the beginning and keep only top 3
          const updatedRepos = [repository, ...filteredRepos].slice(0, 3);

          return {
            recentRepositoriesByScope: {
              ...state.recentRepositoriesByScope,
              [scope]: updatedRepos,
            },
          };
        }),

      clearRecentRepositories: (scope: string) =>
        set((state) => ({
          recentRepositoriesByScope: {
            ...state.recentRepositoriesByScope,
            [scope]: [],
          },
        })),

      getRecentRepositories: (scope: string) =>
        get().recentRepositoriesByScope[scope] ?? EMPTY_RECENT_REPOSITORIES,

      setLastSelectedProvider: (provider: Provider | null) =>
        set(() => ({
          lastSelectedProvider: provider,
        })),

      getLastSelectedProvider: () => get().lastSelectedProvider,
    }),
    {
      name: "home-store", // unique name for localStorage
      storage: createJSONStorage(() => localStorage),
      version: 1,
      // Legacy recents have no owner. Do not assign them to the backend that
      // happens to be active when this browser upgrades.
      migrate: (persistedState) => ({
        ...initialState,
        lastSelectedProvider:
          (persistedState as Partial<HomeState>)?.lastSelectedProvider ?? null,
      }),
    },
  ),
);
