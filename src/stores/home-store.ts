import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import { GitRepository } from "#/types/git";
import { Provider } from "#/types/settings";

interface HomeState {
  recentRepositories: GitRepository[];
  lastSelectedProvider: Provider | null;
}

interface HomeActions {
  addRecentRepository: (repository: GitRepository) => void;
  clearRecentRepositories: () => void;
  getRecentRepositories: () => GitRepository[];
  setLastSelectedProvider: (provider: Provider | null) => void;
  getLastSelectedProvider: () => Provider | null;
}

type HomeStore = HomeState & HomeActions;

const initialState: HomeState = {
  recentRepositories: [],
  lastSelectedProvider: null,
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
    }),
    {
      name: "home-store", // unique name for localStorage
      storage: createJSONStorage(() => localStorage),
    },
  ),
);
