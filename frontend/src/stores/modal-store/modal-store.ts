import { create } from "zustand";
import type {
  ModalConfigMap,
  ModalType,
  OpenModalOptions,
  ModalStore,
} from "./types";

export const useModalStore = create<ModalStore>((set, get) => ({
  modalStack: [],

  isOpen: () => get().modalStack.length > 0,
  topModal: () => get().modalStack.at(-1),

  openModal: <T extends ModalType>(
    type: T,
    props: ModalConfigMap[T],
    options?: OpenModalOptions,
  ) =>
    set((state) => {
      const topModal = state.modalStack.at(-1);

      // Prevent duplicate unless explicitly allowed
      if (!options?.allowDuplicate && topModal?.type === type) {
        return state;
      }

      return {
        modalStack: [...state.modalStack, { type, props }],
      };
    }),

  closeModal: () =>
    set((state) => ({
      modalStack: state.modalStack.slice(0, -1),
    })),

  closeModalByType: (type) =>
    set((state) => ({
      modalStack: state.modalStack.filter((m) => m.type !== type),
    })),

  closeAllModals: () => set({ modalStack: [] }),

  replaceModal: <T extends ModalType>(type: T, props: ModalConfigMap[T]) =>
    set((state) => ({
      modalStack: [...state.modalStack.slice(0, -1), { type, props }],
    })),
}));
