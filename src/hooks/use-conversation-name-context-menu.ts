import React from "react";
import { useTranslation } from "react-i18next";
import { useNavigation } from "#/context/navigation-context";
import { useActiveBackend } from "#/contexts/active-backend-context";
import useMetricsStore from "#/stores/metrics-store";
import { useDeleteConversation } from "./mutation/use-delete-conversation";
import { useUnifiedPauseConversation } from "./mutation/use-unified-stop-conversation";
import { useUpdateConversationPublicFlag } from "./mutation/use-update-conversation-public-flag";
import { useActiveConversation } from "./query/use-active-conversation";
import { useEventStore } from "#/stores/use-event-store";
import { displaySuccessToast } from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";
import { getStoredConversationMetadata } from "#/api/conversation-metadata-store";

import { useDownloadConversation } from "./use-download-conversation";
import {
  adaptSystemMessage,
  SystemMessageForModal,
} from "#/utils/system-message-adapter";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import { isExecutionActive } from "#/utils/status";

interface UseConversationNameContextMenuProps {
  conversationId?: string;
  executionStatus?: ExecutionStatus | null;
  showOptions?: boolean;
  onContextMenuToggle?: (isOpen: boolean) => void;
}

export function useConversationNameContextMenu({
  conversationId,
  executionStatus,
  showOptions = false,
  onContextMenuToggle,
}: UseConversationNameContextMenuProps) {
  const { t } = useTranslation();
  const { conversationId: currentConversationId, navigate } = useNavigation();
  const { backend } = useActiveBackend();
  const events = useEventStore((state) => state.events);
  const { mutate: deleteConversation } = useDeleteConversation();
  const { mutate: stopConversation } = useUnifiedPauseConversation();
  const { mutate: updatePublicFlag } = useUpdateConversationPublicFlag();
  const { data: conversation } = useActiveConversation();
  const metrics = useMetricsStore();

  const [metricsModalVisible, setMetricsModalVisible] = React.useState(false);
  const [systemModalVisible, setSystemModalVisible] = React.useState(false);
  const [skillsModalVisible, setSkillsModalVisible] = React.useState(false);
  const [pluginsModalVisible, setPluginsModalVisible] = React.useState(false);
  const [hooksModalVisible, setHooksModalVisible] = React.useState(false);
  const [confirmDeleteModalVisible, setConfirmDeleteModalVisible] =
    React.useState(false);
  const [confirmStopModalVisible, setConfirmStopModalVisible] =
    React.useState(false);
  const { mutateAsync: downloadConversation } = useDownloadConversation();

  const systemMessage: SystemMessageForModal | null =
    adaptSystemMessage(events);

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
              navigate("/conversations");
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

  const handleDownloadConversation = async (
    event: React.MouseEvent<HTMLButtonElement>,
  ) => {
    event.preventDefault();
    event.stopPropagation();
    if (conversationId) {
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

  const handleShowPlugins = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    setPluginsModalVisible(true);
    onContextMenuToggle?.(false);
  };

  const handleShowHooks = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    setHooksModalVisible(true);
    onContextMenuToggle?.(false);
  };

  const handleTogglePublic = (nextIsPublic?: boolean) => {
    if (conversationId && conversation) {
      updatePublicFlag({
        conversationId,
        isPublic: nextIsPublic ?? !conversation.public,
      });
    }
    // Intentionally don't close the menu — let the user see the toggle flip.
  };

  const shareUrl = React.useMemo(() => {
    if (!conversationId) return "";
    // On cloud backends, the shareable URL must point at the cloud
    // environment's host (e.g. the cloud app domain) rather than the local
    // dev origin so the link works for anyone the user shares it with.
    const origin =
      backend.kind === "cloud"
        ? backend.host.replace(/\/+$/, "")
        : window.location.origin;
    return `${origin}/shared/conversations/${conversationId}`;
  }, [conversationId, backend.kind, backend.host]);

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
    handleDownloadConversation,
    handleDisplayCost,
    handleShowAgentTools,
    handleShowSkills,
    handleShowPlugins,
    handleShowHooks,
    handleTogglePublic,
    handleCopyShareLink,
    shareUrl,
    handleConfirmDelete,
    handleConfirmStop,

    // Modal states
    metricsModalVisible,
    setMetricsModalVisible,
    systemModalVisible,
    setSystemModalVisible,
    skillsModalVisible,
    setSkillsModalVisible,
    pluginsModalVisible,
    setPluginsModalVisible,
    hooksModalVisible,
    setHooksModalVisible,
    confirmDeleteModalVisible,
    setConfirmDeleteModalVisible,
    confirmStopModalVisible,
    setConfirmStopModalVisible,

    // Data
    metrics,
    systemMessage,

    shouldShowStop: isExecutionActive(executionStatus),
    shouldShowDownloadConversation: Boolean(conversationId && showOptions),
    shouldShowDisplayCost: showOptions,
    shouldShowAgentTools: Boolean(showOptions && systemMessage),
    shouldShowSkills: Boolean(showOptions && conversationId),
    shouldShowPlugins: Boolean(
      showOptions &&
      conversationId &&
      (getStoredConversationMetadata(conversationId)?.plugins?.length ?? 0) > 0,
    ),
    shouldShowHooks: Boolean(
      showOptions && conversationId && isExecutionActive(executionStatus),
    ),
  };
}
