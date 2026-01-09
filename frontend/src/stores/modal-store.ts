import { create } from "zustand";
import { SystemMessageForModal } from "#/utils/system-message-adapter";
import { ApiKey, CreateApiKeyResponse } from "#/api/api-keys";
import { Settings } from "#/types/settings";
import {
  LearnThisRepoFormData,
  MicroagentFormData,
} from "#/types/microagent-management";

// Base modal options that all modals can use
interface BaseModalOptions {
  closeOnEscape?: boolean;
  closeOnBackdrop?: boolean;
}

// Integration data type for configure modal
interface IntegrationData {
  id: number;
  keycloak_user_id: string;
  status: string;
  workspace?: {
    id: number;
    name: string;
    status: string;
    editable: boolean;
  };
}

// Core props for each modal type (without BaseModalOptions)
interface ModalCoreProps {
  "confirm-delete": {
    conversationTitle?: string;
    onConfirm: () => void;
  };
  "confirm-stop": {
    onConfirm: () => void;
  };
  "exit-conversation": {
    onConfirm: () => void;
  };
  settings: {
    settings?: Settings;
  };
  feedback: {
    polarity: "positive" | "negative";
  };
  metrics: Record<string, never>; // No props needed, uses store
  "system-message": {
    systemMessage: SystemMessageForModal | null;
  };
  skills: Record<string, never>; // No props needed, uses hooks

  // Generic confirmation modal
  confirmation: {
    text: string;
    onConfirm: () => void;
  };

  // API Key modals
  "create-api-key": {
    onKeyCreated: (newKey: CreateApiKeyResponse) => void;
  };
  "delete-api-key": {
    keyToDelete: ApiKey | null;
    onDeleted: () => void;
  };
  "new-api-key": {
    newlyCreatedKey: CreateApiKeyResponse | null;
  };

  // Microagent modals
  "launch-microagent": {
    eventId: number;
    selectedRepo: string;
    onLaunch: (query: string, target: string, triggers: string[]) => void;
    isLoading: boolean;
  };
  "learn-this-repo": {
    onConfirm: (formData: LearnThisRepoFormData) => void;
    isLoading: boolean;
  };
  "upsert-microagent": {
    onConfirm: (formData: MicroagentFormData) => void;
    isLoading: boolean;
    isUpdate?: boolean;
  };

  // End session confirmation (used by settings form)
  "end-session": {
    onConfirm: () => void;
  };

  // Integration configure modal
  "configure-integration": {
    platform: "jira" | "jira-dc" | "linear";
    platformName: string;
    integrationData?: IntegrationData;
    onConfirm: (data: {
      workspace: string;
      webhookSecret: string;
      serviceAccountEmail: string;
      serviceAccountApiKey: string;
      isActive: boolean;
    }) => void;
    onLink: (workspace: string) => void;
    onUnlink: () => void;
  };

  // Auth modals
  reauth: Record<string, never>; // No props needed
  "email-verification": {
    userId?: string | null;
  };

  // Analytics consent modal
  "analytics-consent": Record<string, never>; // Uses form submission internally

  // Payment modals
  "setup-payment": Record<string, never>; // No props needed
  "cancel-subscription": {
    endDate?: string;
  };
}

// Final config map: each modal's props automatically include BaseModalOptions
export type ModalConfigMap = {
  [K in keyof ModalCoreProps]: ModalCoreProps[K] & BaseModalOptions;
};

export type ModalType = keyof ModalConfigMap;

export interface ModalInstance<T extends ModalType = ModalType> {
  type: T;
  props: ModalConfigMap[T];
}

interface OpenModalOptions {
  allowDuplicate?: boolean;
}

interface ModalState {
  modalStack: ModalInstance[];
}

interface ModalSelectors {
  isOpen: () => boolean;
  topModal: () => ModalInstance | undefined;
}

interface ModalActions {
  openModal: <T extends ModalType>(
    type: T,
    props: ModalConfigMap[T],
    options?: OpenModalOptions,
  ) => void;
  closeModal: () => void;
  closeModalByType: (type: ModalType) => void;
  closeAllModals: () => void;
  replaceModal: <T extends ModalType>(
    type: T,
    props: ModalConfigMap[T],
  ) => void;
}

type ModalStore = ModalState & ModalSelectors & ModalActions;

export const useModalStore = create<ModalStore>((set, get) => ({
  modalStack: [],

  // Selectors
  isOpen: () => get().modalStack.length > 0,
  topModal: () => get().modalStack.at(-1),

  // Actions
  openModal: (type, props, options) =>
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

  replaceModal: (type, props) =>
    set((state) => ({
      modalStack: [...state.modalStack.slice(0, -1), { type, props }],
    })),
}));
