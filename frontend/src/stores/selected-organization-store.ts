import { create } from "zustand";
import { devtools } from "zustand/middleware";

interface SelectedOrganizationState {
  organizationId: string | null;
}

interface SelectedOrganizationActions {
  setOrganizationId: (orgId: string | null) => void;
}

type SelectedOrganizationStore = SelectedOrganizationState &
  SelectedOrganizationActions;

const initialState: SelectedOrganizationState = {
  organizationId: null,
};

export const useSelectedOrganizationStore = create<SelectedOrganizationStore>()(
  devtools(
    (set) => ({
      ...initialState,
      setOrganizationId: (organizationId) => set({ organizationId }),
    }),
    { name: "SelectedOrganizationStore" },
  ),
);

export const getSelectedOrganizationIdFromStore = (): string | null =>
  useSelectedOrganizationStore.getState().organizationId;
