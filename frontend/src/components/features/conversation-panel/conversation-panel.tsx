import React from "react";
import { NavLink, useParams, useNavigate } from "react-router";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { usePaginatedConversations } from "#/hooks/query/use-paginated-conversations";
import { useStartTasks } from "#/hooks/query/use-start-tasks";
import { useInfiniteScroll } from "#/hooks/use-infinite-scroll";
import { useDeleteConversation } from "#/hooks/mutation/use-delete-conversation";
import { useUnifiedPauseConversationSandbox } from "#/hooks/mutation/use-unified-stop-conversation";
import { ConfirmDeleteModal } from "./confirm-delete-modal";
import { ConfirmStopModal } from "./confirm-stop-modal";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ExitConversationModal } from "./exit-conversation-modal";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { Provider } from "#/types/settings";
import { useUpdateConversation } from "#/hooks/mutation/use-update-conversation";
import { displaySuccessToast } from "#/utils/custom-toast-handlers";
import { ConversationCard } from "./conversation-card/conversation-card";
import { StartTaskCard } from "./start-task-card/start-task-card";

interface ConversationPanelProps {
  onClose: () => void;
}

export function ConversationPanel({ onClose }: ConversationPanelProps) {
  const { t } = useTranslation();
  const { conversationId: currentConversationId } = useParams();
  const ref = useClickOutsideElement<HTMLDivElement>(onClose);
  const navigate = useNavigate();

  const [confirmDeleteModalVisible, setConfirmDeleteModalVisible] =
    React.useState(false);
  const [confirmStopModalVisible, setConfirmStopModalVisible] =
    React.useState(false);
  const [
    confirmExitConversationModalVisible,
    setConfirmExitConversationModalVisible,
  ] = React.useState(false);
  const [selectedConversationId, setSelectedConversationId] = React.useState<
    string | null
  >(null);
  const [selectedConversationVersion, setSelectedConversationVersion] =
    React.useState<"V0" | "V1" | undefined>(undefined);
  const [openContextMenuId, setOpenContextMenuId] = React.useState<
    string | null
  >(null);

  const {
    data,
    isFetching,
    error,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
  } = usePaginatedConversations();

  // Fetch in-progress start tasks
  const { data: startTasks } = useStartTasks();

  // Flatten all pages into a single array of conversations
  const conversations = data?.pages.flatMap((page) => page.results) ?? [];

  const { mutate: deleteConversation } = useDeleteConversation();
  const { mutate: pauseConversationSandbox } =
    useUnifiedPauseConversationSandbox();
  const { mutate: updateConversation } = useUpdateConversation();

  // Set up infinite scroll
  const scrollContainerRef = useInfiniteScroll({
    hasNextPage: !!hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
    threshold: 200, // Load more when 200px from bottom
  });

  const handleDeleteProject = (conversationId: string) => {
    setConfirmDeleteModalVisible(true);
    setSelectedConversationId(conversationId);
  };

  const handleStopConversation = (
    conversationId: string,
    version?: "V0" | "V1",
  ) => {
    setConfirmStopModalVisible(true);
    setSelectedConversationId(conversationId);
    setSelectedConversationVersion(version);
  };

  const handleConversationTitleChange = async (
    conversationId: string,
    newTitle: string,
  ) => {
    updateConversation(
      { conversationId, newTitle },
      {
        onSuccess: () => {
          displaySuccessToast(t(I18nKey.CONVERSATION$TITLE_UPDATED));
        },
      },
    );
  };

  const handleConfirmDelete = () => {
    if (selectedConversationId) {
      deleteConversation(
        { conversationId: selectedConversationId },
        {
          onSuccess: () => {
            if (selectedConversationId === currentConversationId) {
              navigate("/");
            }
          },
        },
      );
    }
  };

  const handleConfirmStop = () => {
    if (selectedConversationId) {
      pauseConversationSandbox({
        conversationId: selectedConversationId,
        version: selectedConversationVersion,
      });
    }
  };

  return (
    <div
      ref={(node) => {
        // TODO: Combine both refs somehow
        if (ref.current !== node) ref.current = node;
        if (scrollContainerRef.current !== node)
          scrollContainerRef.current = node;
      }}
      data-testid="conversation-panel"
      className="w-full md:w-[400px] h-full border border-[#525252] bg-[#25272D] rounded-lg overflow-y-auto absolute custom-scrollbar-always"
    >
      {isFetching && conversations.length === 0 && (
        <div className="w-full h-full absolute flex justify-center items-center">
          <LoadingSpinner size="small" />
        </div>
      )}
      {error && (
        <div className="flex flex-col items-center justify-center h-full">
          <p className="text-danger">{error.message}</p>
        </div>
      )}
      {!isFetching && conversations?.length === 0 && !startTasks?.length && (
        <div className="flex flex-col items-center justify-center h-full">
          <p className="text-neutral-400">
            {t(I18nKey.CONVERSATION$NO_CONVERSATIONS)}
          </p>
        </div>
      )}
      {/* Render in-progress start tasks first */}
      {startTasks?.map((task) => (
        <NavLink
          key={task.id}
          to={`/conversations/task-${task.id}`}
          onClick={onClose}
        >
          <StartTaskCard task={task} />
        </NavLink>
      ))}
      {/* Then render completed conversations */}
      {conversations?.map((project) => (
        <NavLink
          key={project.conversation_id}
          to={`/conversations/${project.conversation_id}`}
          onClick={onClose}
        >
          <ConversationCard
            onDelete={() => handleDeleteProject(project.conversation_id)}
            onStop={() =>
              handleStopConversation(
                project.conversation_id,
                project.conversation_version,
              )
            }
            onChangeTitle={(title) =>
              handleConversationTitleChange(project.conversation_id, title)
            }
            title={project.title}
            selectedRepository={{
              selected_repository: project.selected_repository,
              selected_branch: project.selected_branch,
              git_provider: project.git_provider as Provider,
            }}
            lastUpdatedAt={project.last_updated_at}
            createdAt={project.created_at}
            conversationStatus={project.status}
            conversationId={project.conversation_id}
            conversationVersion={project.conversation_version}
            contextMenuOpen={openContextMenuId === project.conversation_id}
            onContextMenuToggle={(isOpen) =>
              setOpenContextMenuId(isOpen ? project.conversation_id : null)
            }
          />
        </NavLink>
      ))}

      {/* Loading indicator for fetching more conversations */}
      {isFetchingNextPage && (
        <div className="flex justify-center py-4">
          <LoadingSpinner size="small" />
        </div>
      )}

      {confirmDeleteModalVisible && (
        <ConfirmDeleteModal
          onConfirm={() => {
            handleConfirmDelete();
            setConfirmDeleteModalVisible(false);
          }}
          onCancel={() => setConfirmDeleteModalVisible(false)}
        />
      )}

      {confirmStopModalVisible && (
        <ConfirmStopModal
          onConfirm={() => {
            handleConfirmStop();
            setConfirmStopModalVisible(false);
          }}
          onCancel={() => setConfirmStopModalVisible(false)}
        />
      )}

      {confirmExitConversationModalVisible && (
        <ExitConversationModal
          onConfirm={() => {
            onClose();
          }}
          onClose={() => setConfirmExitConversationModalVisible(false)}
        />
      )}
    </div>
  );
}
