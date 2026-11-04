import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import { Branch, GitRepository } from "#/types/git";
import { Provider } from "#/types/settings";

export interface HomeRepositorySelection {
  repository: GitRepository;
  branch: Branch;
  provider: Provider;
}

interface HomeState {
  recentRepositories: GitRepository[];
  lastSelectedProvider: Provider | null;
  /** The home launcher's repository selection, keyed by backend id. */
  repositorySelections: Record<string, HomeRepositorySelection>;
}

interface HomeActions {
  addRecentRepository: (repository: GitRepository) => void;
  clearRecentRepositories: () => void;
  getRecentRepositories: () => GitRepository[];
  setLastSelectedProvider: (provider: Provider | null) => void;
  getLastSelectedProvider: () => Provider | null;
  setRepositorySelection: (
    backendId: string,
    selection: HomeRepositorySelection,
  ) => void;
  clearRepositorySelection: (backendId: string) => void;
}

type HomeStore = HomeState & HomeActions;

const initialState: HomeState = {
  recentRepositories: [],
  lastSelectedProvider: null,
  repositorySelections: {},
};

export const useHomeStore = create<HomeStore>()(
  persist(
    (set, get) => ({
      ...initialState,

      addRecentRepository: (repository: GitRepository) =>
        set((state) => {
          // Remove the repository if it already exists to avoid duplicates
          const filteredRepos = state.recentRepositories.filter(
            (repo) => repo.id !== repository.id,
          );

          // Add the new repository to the beginning and keep only top 3
          const updatedRepos = [repository, ...filteredRepos].slice(0, 3);

          return {
            recentRepositories: updatedRepos,
          };
        }),

      clearRecentRepositories: () =>
        set(() => ({
          recentRepositories: [],
        })),

      getRecentRepositories: () => get().recentRepositories,

      setLastSelectedProvider: (provider: Provider | null) =>
        set(() => ({
          lastSelectedProvider: provider,
        })),

      getLastSelectedProvider: () => get().lastSelectedProvider,

      setRepositorySelection: (backendId, selection) =>
        set((state) => ({
          repositorySelections: {
            ...state.repositorySelections,
            [backendId]: selection,
          },
        })),

      clearRepositorySelection: (backendId) =>
        set((state) => {
          if (!(backendId in state.repositorySelections)) return state;
          const { [backendId]: _cleared, ...rest } = state.repositorySelections;
          return { repositorySelections: rest };
        }),
    }),
    {
      name: "home-store", // unique name for localStorage
      storage: createJSONStorage(() => localStorage),
    },
  ),
);
