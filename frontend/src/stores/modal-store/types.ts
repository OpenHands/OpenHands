import { SystemMessageForModal } from "#/utils/system-message-adapter";
import { ApiKey, CreateApiKeyResponse } from "#/api/api-keys";
import { Settings } from "#/types/settings";
import {
  LearnThisRepoFormData,
  MicroagentFormData,
} from "#/types/microagent-management";

// Base modal options that all modals can use
export interface BaseModalOptions {
  closeOnEscape?: boolean;
  closeOnBackdrop?: boolean;
}

export interface IntegrationData {
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

// Core props for each modal type
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
  "end-session": {
    onConfirm: () => void;
  };
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
  reauth: Record<string, never>; // No props needed
  "email-verification": {
    userId?: string | null;
  };
  "analytics-consent": Record<string, never>; // Uses form submission internally
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

export interface OpenModalOptions {
  allowDuplicate?: boolean;
}

export interface ModalState {
  modalStack: ModalInstance[];
}

export interface ModalSelectors {
  isOpen: () => boolean;
  topModal: () => ModalInstance | undefined;
}

export interface ModalActions {
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

export type ModalStore = ModalState & ModalSelectors & ModalActions;
