import { create } from "zustand";
import { persist, createJSONStorage, devtools } from "zustand/middleware";

interface SelectedOrganizationState {
  orgId: string | null;
}

interface SelectedOrganizationActions {
  setOrgId: (orgId: string | null) => void;
}

type SelectedOrganizationStore = SelectedOrganizationState &
  SelectedOrganizationActions;

const initialState: SelectedOrganizationState = {
  orgId: null,
};

export const useSelectedOrganizationStore = create<SelectedOrganizationStore>()(
  devtools(
    persist(
      (set) => ({
        ...initialState,
        setOrgId: (orgId) => set({ orgId }),
      }),
      {
        name: "selected-organization-id",
        storage: createJSONStorage(() => localStorage),
      },
    ),
    { name: "SelectedOrganizationStore" },
  ),
);

export const getSelectedOrgIdFromStore = (): string | null =>
  useSelectedOrganizationStore.getState().orgId;
