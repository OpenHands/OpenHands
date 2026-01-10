import React from "react";
import { useTranslation } from "react-i18next";
import { usePostHog } from "posthog-js/react";
import { useParams, useNavigate } from "react-router";
import { transformVSCodeUrl } from "#/utils/vscode-url-helper";
import useMetricsStore from "#/stores/metrics-store";
import { ConversationStatus } from "#/types/conversation-status";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { useDeleteConversation } from "./mutation/use-delete-conversation";
import { useUnifiedPauseConversationSandbox } from "./mutation/use-unified-stop-conversation";
import { useGetTrajectory } from "./mutation/use-get-trajectory";
import { useUpdateConversationPublicFlag } from "./mutation/use-update-conversation-public-flag";
import { downloadTrajectory } from "#/utils/download-trajectory";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";
import { useEventStore } from "#/stores/use-event-store";

import { useActiveConversation } from "./query/use-active-conversation";
import { useDownloadConversation } from "./use-download-conversation";
import {
  adaptSystemMessage,
  SystemMessageForModal,
} from "#/utils/system-message-adapter";
import { useModalStore } from "#/stores/modal-store";

interface UseConversationNameContextMenuProps {
  conversationId?: string;
  conversationStatus?: ConversationStatus;
  showOptions?: boolean;
  onContextMenuToggle?: (isOpen: boolean) => void;
}

export function useConversationNameContextMenu({
  conversationId,
  conversationStatus = "STOPPED",
  showOptions = false,
  onContextMenuToggle,
}: UseConversationNameContextMenuProps) {
  const posthog = usePostHog();
  const { t } = useTranslation();
  const { conversationId: currentConversationId } = useParams();
  const navigate = useNavigate();
  const events = useEventStore((state) => state.events);
  const { mutate: deleteConversation } = useDeleteConversation();
  const { mutate: stopConversation } = useUnifiedPauseConversationSandbox();
  const { mutate: getTrajectory } = useGetTrajectory();
  const { mutate: updatePublicFlag } = useUpdateConversationPublicFlag();
  const { data: conversation } = useActiveConversation();
  const metrics = useMetricsStore();

  const openModal = useModalStore((state) => state.openModal);
  const closeModal = useModalStore((state) => state.closeModal);
  const { mutateAsync: downloadConversation } = useDownloadConversation();

  const systemMessage: SystemMessageForModal | null =
    adaptSystemMessage(events);

  const handleDelete = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    openModal("confirm-delete", {
      conversationTitle: conversation?.title,
      onConfirm: () => {
        if (conversationId) {
          deleteConversation(
            { conversationId },
            {
              onSuccess: () => {
                if (conversationId === currentConversationId) {
                  navigate("/");
                }
              },
            },
          );
        }
        closeModal();
      },
    });
    onContextMenuToggle?.(false);
  };

  const handleStop = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    openModal("confirm-stop", {
      onConfirm: () => {
        if (conversationId) {
          stopConversation({ conversationId });
        }
        closeModal();
      },
    });
    onContextMenuToggle?.(false);
  };

  const handleEdit = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    // This will be handled by the parent component to switch to edit mode
    onContextMenuToggle?.(false);
  };

  const handleExportConversation = (
    event: React.MouseEvent<HTMLButtonElement>,
  ) => {
    event.preventDefault();
    event.stopPropagation();

    if (!conversationId) {
      displayErrorToast(t(I18nKey.CONVERSATION$DOWNLOAD_ERROR));
      return;
    }

    getTrajectory(conversationId, {
      onSuccess: async (data) => {
        await downloadTrajectory(
          conversationId ?? t(I18nKey.CONVERSATION$UNKNOWN),
          data.trajectory,
        );
      },
      onError: () => {
        displayErrorToast(t(I18nKey.CONVERSATION$DOWNLOAD_ERROR));
      },
    });

    onContextMenuToggle?.(false);
  };

  const handleDownloadViaVSCode = async (
    event: React.MouseEvent<HTMLButtonElement>,
  ) => {
    event.preventDefault();
    event.stopPropagation();
    posthog.capture("download_via_vscode_button_clicked");

    // Fetch the VS Code URL from the API
    if (conversationId) {
      try {
        const data = await ConversationService.getVSCodeUrl(conversationId);
        if (data.vscode_url) {
          const transformedUrl = transformVSCodeUrl(data.vscode_url);
          if (transformedUrl) {
            window.open(transformedUrl, "_blank");
          }
        }
        // VS Code URL not available
      } catch {
        // Failed to fetch VS Code URL
      }
    }

    onContextMenuToggle?.(false);
  };

  const handleDownloadConversation = async (
    event: React.MouseEvent<HTMLButtonElement>,
  ) => {
    event.preventDefault();
    event.stopPropagation();
    if (conversationId && conversation?.conversation_version === "V1") {
      await downloadConversation(conversationId);
    }
    onContextMenuToggle?.(false);
  };

  const handleDisplayCost = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    openModal("metrics", {});
    onContextMenuToggle?.(false);
  };

  const handleShowAgentTools = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    openModal("system-message", { systemMessage });
    onContextMenuToggle?.(false);
  };

  const handleShowSkills = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    openModal("skills", {});
    onContextMenuToggle?.(false);
  };

  const handleTogglePublic = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();

    if (conversationId && conversation) {
      // Toggle the current public state
      const newPublicState = !conversation.public;
      updatePublicFlag({
        conversationId,
        isPublic: newPublicState,
      });
    }
    // Don't close menu - let user see the toggle state change
  };

  const shareUrl = React.useMemo(() => {
    if (conversationId) {
      return `${window.location.origin}/shared/conversations/${conversationId}`;
    }
    return "";
  }, [conversationId]);

  const handleCopyShareLink = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();

    if (!shareUrl) {
      onContextMenuToggle?.(false);
      return;
    }

    navigator.clipboard.writeText(shareUrl);
    displaySuccessToast(t(I18nKey.CONVERSATION$LINK_COPIED));
  };

  return {
    // Handlers
    handleDelete,
    handleStop,
    handleEdit,
    handleExportConversation,
    handleDownloadViaVSCode,
    handleDownloadConversation,
    handleDisplayCost,
    handleShowAgentTools,
    handleShowSkills,
    handleTogglePublic,
    handleCopyShareLink,
    shareUrl,

    // Data
    metrics,
    systemMessage,

    // Computed values for conditional rendering
    shouldShowStop: conversationStatus !== "STOPPED",
    shouldShowDownload: Boolean(conversationId && showOptions),
    shouldShowExport: Boolean(conversationId && showOptions),
    shouldShowDownloadConversation: Boolean(
      conversationId &&
      showOptions &&
      conversation?.conversation_version === "V1",
    ),
    shouldShowDisplayCost: showOptions,
    shouldShowAgentTools: Boolean(showOptions && systemMessage),
    shouldShowSkills: Boolean(showOptions && conversationId),
  };
}
