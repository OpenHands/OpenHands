export interface BaseModalOptions {
  // Whether ESC key closes the modal. Default: true
  closeOnEscape?: boolean;
  // Whether clicking backdrop closes the modal. Default: true
  closeOnBackdrop?: boolean;
}

// Core props for each modal type.
interface ModalCoreProps {
  "confirm-delete": {
    conversationTitle?: string;
    onConfirm: () => void;
  };
  "confirm-stop": {
    onConfirm: () => void;
  };
  feedback: {
    polarity: "positive" | "negative";
  };
}

/**
 * Final config map: each modal's props automatically include BaseModalOptions.
 * This allows any modal to optionally configure ESC/backdrop behavior.
 */
export type ModalConfigMap = {
  [K in keyof ModalCoreProps]: ModalCoreProps[K] & BaseModalOptions;
};

// Union of all modal type names
export type ModalType = keyof ModalConfigMap;

// A single modal instance in the stack
export interface ModalInstance<T extends ModalType = ModalType> {
  type: T;
  props: ModalConfigMap[T];
}

// Options when opening a modal
export interface OpenModalOptions {
  // Allow opening duplicate of same modal type.
  allowDuplicate?: boolean;
}

export interface ModalState {
  // Stack of currently open modals (last = topmost)
  modalStack: ModalInstance[];
}

export interface ModalSelectors {
  // Returns true if any modal is open
  isOpen: () => boolean;
  // Returns the topmost modal in the stack
  topModal: () => ModalInstance | undefined;
}

export interface ModalActions {
  openModal: <T extends ModalType>(
    type: T,
    props: ModalConfigMap[T],
    options?: OpenModalOptions,
  ) => void;

  // Close the topmost modal
  closeModal: () => void;

  // Close a specific modal by type
  closeModalByType: (type: ModalType) => void;

  // Close all open modals
  closeAllModals: () => void;

  // Replace the topmost modal with a new one
  replaceModal: <T extends ModalType>(
    type: T,
    props: ModalConfigMap[T],
  ) => void;
}

// Complete modal store type
export type ModalStore = ModalState & ModalSelectors & ModalActions;
