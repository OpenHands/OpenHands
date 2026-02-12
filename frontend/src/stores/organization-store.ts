import { create } from "zustand";
import { devtools } from "zustand/middleware";

interface OrganizationState {
  shouldHideSelector: boolean;
}

interface OrganizationActions {
  setShouldHideSelector: (value: boolean) => void;
}

type OrganizationStore = OrganizationState & OrganizationActions;

const initialState: OrganizationState = {
  shouldHideSelector: false,
};

export const useOrganizationStore = create<OrganizationStore>()(
  devtools(
    (set) => ({
      ...initialState,
      setShouldHideSelector: (shouldHideSelector) =>
        set({ shouldHideSelector }),
    }),
    { name: "OrganizationStore" },
  ),
);
