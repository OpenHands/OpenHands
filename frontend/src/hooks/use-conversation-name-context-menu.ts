import { useTranslation } from "react-i18next";
import React from "react";
import { usePostHog } from "posthog-js/react";
import { useParams, useNavigate } from "react-router";
import { transformVSCodeUrl } from "#/utils/vscode-url-helper";
import useMetricsStore from "#/stores/metrics-store";
import { isSystemMessage, isActionOrObservation } from "#/types/core/guards";
import { ConversationStatus } from "#/types/conversation-status";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { useDeleteConversation } from "./mutation/use-delete-conversation";
import { useUnifiedPauseConversationSandbox } from "./mutation/use-unified-stop-conversation";
import { useGetTrajectory } from "./mutation/use-get-trajectory";
import { downloadTrajectory } from "#/utils/download-trajectory";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";
import { useEventStore } from "#/stores/use-event-store";
import { isV0Event } from "#/types/v1/type-guards";
import { useActiveConversation } from "./query/use-active-conversation";
import { useDownloadConversation } from "./use-download-conversation";

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
  const { data: conversation } = useActiveConversation();
  const { mutate: deleteConversation } = useDeleteConversation();
  const { mutate: stopConversation } = useUnifiedPauseConversationSandbox();
  const { mutate: getTrajectory } = useGetTrajectory();
  const metrics = useMetricsStore();

  const [metricsModalVisible, setMetricsModalVisible] = React.useState(false);
  const [systemModalVisible, setSystemModalVisible] = React.useState(false);
  const [skillsModalVisible, setSkillsModalVisible] = React.useState(false);
  const [confirmDeleteModalVisible, setConfirmDeleteModalVisible] =
    React.useState(false);
  const [confirmStopModalVisible, setConfirmStopModalVisible] =
    React.useState(false);
  const { mutateAsync: downloadConversation } = useDownloadConversation();

  const systemMessage = events
    .filter(isV0Event)
    .filter(isActionOrObservation)
    .find(isSystemMessage);

  const handleDelete = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setConfirmDeleteModalVisible(true);
    onContextMenuToggle?.(false);
  };

  const handleStop = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setConfirmStopModalVisible(true);
    onContextMenuToggle?.(false);
  };

  const handleConfirmDelete = () => {
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
    setConfirmDeleteModalVisible(false);
  };

  const handleConfirmStop = () => {
    if (conversationId) {
      stopConversation({ conversationId });
    }
    setConfirmStopModalVisible(false);
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
    setMetricsModalVisible(true);
    onContextMenuToggle?.(false);
  };

  const handleShowAgentTools = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    setSystemModalVisible(true);
    onContextMenuToggle?.(false);
  };

  const handleShowSkills = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    setSkillsModalVisible(true);
    onContextMenuToggle?.(false);
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
    handleConfirmDelete,
    handleConfirmStop,

    // Modal states
    metricsModalVisible,
    setMetricsModalVisible,
    systemModalVisible,
    setSystemModalVisible,
    skillsModalVisible,
    setSkillsModalVisible,
    confirmDeleteModalVisible,
    setConfirmDeleteModalVisible,
    confirmStopModalVisible,
    setConfirmStopModalVisible,

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
