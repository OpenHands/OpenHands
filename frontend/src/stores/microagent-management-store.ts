import { create } from "zustand";
import { GitRepository } from "#/types/git";
import { IMicroagentItem } from "#/types/microagent-management";

interface MicroagentManagementState {
  // Modal visibility states
  addMicroagentModalVisible: boolean;
  updateMicroagentModalVisible: boolean;
  learnThisRepoModalVisible: boolean;

  // Repository states
  selectedRepository: GitRepository | null;
  personalRepositories: GitRepository[];
  organizationRepositories: GitRepository[];
  repositories: GitRepository[];

  // Microagent states
  selectedMicroagentItem: IMicroagentItem | null;
}

interface MicroagentManagementActions {
  // Modal actions
  setAddMicroagentModalVisible: (visible: boolean) => void;
  setUpdateMicroagentModalVisible: (visible: boolean) => void;
  setLearnThisRepoModalVisible: (visible: boolean) => void;

  // Repository actions
  setSelectedRepository: (repository: GitRepository | null) => void;
  setPersonalRepositories: (repositories: GitRepository[]) => void;
  setOrganizationRepositories: (repositories: GitRepository[]) => void;
  setRepositories: (repositories: GitRepository[]) => void;

  // Microagent actions
  setSelectedMicroagentItem: (item: IMicroagentItem | null) => void;
}

type MicroagentManagementStore = MicroagentManagementState &
  MicroagentManagementActions;

export const useMicroagentManagementStore = create<MicroagentManagementStore>(
  (set) => ({
    // Initial state
    addMicroagentModalVisible: false,
    updateMicroagentModalVisible: false,
    learnThisRepoModalVisible: false,
    selectedRepository: null,
    personalRepositories: [],
    organizationRepositories: [],
    repositories: [],
    selectedMicroagentItem: null,

    // Actions
    setAddMicroagentModalVisible: (visible: boolean) =>
      set({ addMicroagentModalVisible: visible }),

    setUpdateMicroagentModalVisible: (visible: boolean) =>
      set({ updateMicroagentModalVisible: visible }),

    setLearnThisRepoModalVisible: (visible: boolean) =>
      set({ learnThisRepoModalVisible: visible }),

    setSelectedRepository: (repository: GitRepository | null) =>
      set({ selectedRepository: repository }),

    setPersonalRepositories: (repositories: GitRepository[]) =>
      set({ personalRepositories: repositories }),

    setOrganizationRepositories: (repositories: GitRepository[]) =>
      set({ organizationRepositories: repositories }),

    setRepositories: (repositories: GitRepository[]) => set({ repositories }),

    setSelectedMicroagentItem: (item: IMicroagentItem | null) =>
      set({ selectedMicroagentItem: item }),
  }),
);
