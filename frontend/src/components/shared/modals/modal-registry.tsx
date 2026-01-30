import React from "react";
import type { ModalConfigMap } from "#/stores/modal-store";
import { ConfirmDeleteModal } from "#/components/features/conversation-panel/confirm-delete-modal";
import { ConfirmStopModal } from "#/components/features/conversation-panel/confirm-stop-modal";
import { FeedbackModal } from "#/components/features/feedback/feedback-modal";

/**
 * Modal registry — maps each modal type to a renderer function.
 * To add a new modal, register it here instead of adding a switch case.
 */
export const MODAL_REGISTRY: {
  [K in keyof ModalConfigMap]: (
    props: ModalConfigMap[K],
    onClose: () => void,
  ) => React.ReactNode;
} = {
  "confirm-delete": (props, onClose) => (
    <ConfirmDeleteModal
      conversationTitle={props.conversationTitle}
      onConfirm={() => {
        props.onConfirm();
        onClose();
      }}
      onCancel={onClose}
    />
  ),
  "confirm-stop": (props, onClose) => (
    <ConfirmStopModal
      onConfirm={() => {
        props.onConfirm();
        onClose();
      }}
      onCancel={onClose}
    />
  ),
  feedback: (props, onClose) => (
    <FeedbackModal polarity={props.polarity} onClose={onClose} />
  ),
};
