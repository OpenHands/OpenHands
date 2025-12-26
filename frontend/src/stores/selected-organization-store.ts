import { create } from "zustand";
import { devtools } from "zustand/middleware";

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
    (set) => ({
      ...initialState,
      setOrgId: (orgId) => set({ orgId }),
    }),
    { name: "SelectedOrganizationStore" },
  ),
);

export const getSelectedOrganizationIdFromStore = (): string | null =>
  useSelectedOrganizationStore.getState().orgId;
