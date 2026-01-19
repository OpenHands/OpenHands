import { create } from "zustand";
import type {
  ModalConfigMap,
  ModalType,
  OpenModalOptions,
  ModalStore,
} from "./types";

// Centralized Modal Store
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
      // Prevent duplicate modals of same type unless explicitly allowed
      if (
        !options?.allowDuplicate &&
        state.modalStack.some((m) => m.type === type)
      ) {
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
    set((state) => {
      if (state.modalStack.length === 0) return state;
      return {
        modalStack: [...state.modalStack.slice(0, -1), { type, props }],
      };
    }),
}));
