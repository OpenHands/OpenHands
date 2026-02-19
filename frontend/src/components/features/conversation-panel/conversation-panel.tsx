import React from "react";
import { NavLink, useParams, useNavigate } from "react-router";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { usePaginatedConversations } from "#/hooks/query/use-paginated-conversations";
import { useStartTasks } from "#/hooks/query/use-start-tasks";
import { useInfiniteScroll } from "#/hooks/use-infinite-scroll";
import { useDeleteConversation } from "#/hooks/mutation/use-delete-conversation";
import { useDeleteConversations } from "#/hooks/mutation/use-delete-conversations";
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
import { ConversationCardSkeleton } from "./conversation-card/conversation-card-skeleton";

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
  const [selectedConversationTitle, setSelectedConversationTitle] =
    React.useState<string | null>(null);
  const [selectedConversationVersion, setSelectedConversationVersion] =
    React.useState<"V0" | "V1" | undefined>(undefined);
  const [openContextMenuId, setOpenContextMenuId] = React.useState<
    string | null
  >(null);

  const [selectionMode, setSelectionMode] = React.useState(false);
  const [selectedIds, setSelectedIds] = React.useState<Set<string>>(new Set());

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
  const { mutate: deleteConversations } = useDeleteConversations();
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

  const handleDeleteProject = (conversationId: string, title: string) => {
    setConfirmDeleteModalVisible(true);
    setSelectedConversationId(conversationId);
    setSelectedConversationTitle(title);
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

  const toggleSelectionMode = () => {
    setSelectionMode((prev) => !prev);
    setSelectedIds(new Set());
  };

  const toggleSelection = (conversationId: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(conversationId)) {
        next.delete(conversationId);
      } else {
        next.add(conversationId);
      }
      return next;
    });
  };

  const selectAll = () => {
    if (selectedIds.size === conversations.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(conversations.map((c) => c.conversation_id)));
    }
  };

  const handleBulkDelete = () => {
    setConfirmDeleteModalVisible(true);
  };

  const handleConfirmBulkDelete = () => {
    const ids = Array.from(selectedIds);
    deleteConversations(
      { conversationIds: ids },
      {
        onSuccess: () => {
          if (currentConversationId && selectedIds.has(currentConversationId)) {
            navigate("/");
          }
          setSelectedIds(new Set());
          setSelectionMode(false);
        },
      },
    );
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
      {/* Bulk action bar */}
      {conversations.length > 0 && (
        <div className="sticky top-0 z-10 bg-[#25272D] border-b border-neutral-600 px-3.5 py-2 flex items-center justify-between">
          {selectionMode ? (
            <>
              <label className="flex items-center gap-2 text-sm text-neutral-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={
                    conversations.length > 0 &&
                    selectedIds.size === conversations.length
                  }
                  onChange={selectAll}
                  className="h-4 w-4 accent-blue-500 cursor-pointer"
                  data-testid="select-all-checkbox"
                />
                {t(I18nKey.CONVERSATION$SELECT_ALL)}
              </label>
              <div className="flex items-center gap-2">
                {selectedIds.size > 0 && (
                  <button
                    type="button"
                    onClick={handleBulkDelete}
                    className="text-sm text-danger hover:text-red-400 transition-colors"
                    data-testid="bulk-delete-button"
                  >
                    {t(I18nKey.CONVERSATION$DELETE_SELECTED, {
                      count: selectedIds.size,
                    })}
                  </button>
                )}
                <button
                  type="button"
                  onClick={toggleSelectionMode}
                  className="text-sm text-neutral-400 hover:text-neutral-200 transition-colors"
                  data-testid="cancel-selection-button"
                >
                  {t(I18nKey.BUTTON$CANCEL)}
                </button>
              </div>
            </>
          ) : (
            <button
              type="button"
              onClick={toggleSelectionMode}
              className="text-sm text-neutral-400 hover:text-neutral-200 transition-colors ml-auto"
              data-testid="enter-selection-mode-button"
            >
              {t(I18nKey.CONVERSATION$SELECT)}
            </button>
          )}
        </div>
      )}

      {isFetching && conversations.length === 0 && (
        <div className="space-y-2">
          {Array.from({ length: 5 }).map((_, index) => (
            <ConversationCardSkeleton key={index} />
          ))}
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
      {conversations?.map((project) => {
        const cardContent = (
          <ConversationCard
            onDelete={() =>
              handleDeleteProject(project.conversation_id, project.title)
            }
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
            selectionMode={selectionMode}
            isSelected={selectedIds.has(project.conversation_id)}
            onSelectionToggle={() => toggleSelection(project.conversation_id)}
          />
        );

        if (selectionMode) {
          return <div key={project.conversation_id}>{cardContent}</div>;
        }

        return (
          <NavLink
            key={project.conversation_id}
            to={`/conversations/${project.conversation_id}`}
            onClick={onClose}
          >
            {cardContent}
          </NavLink>
        );
      })}

      {/* Loading indicator for fetching more conversations */}
      {isFetchingNextPage && (
        <div className="flex justify-center py-4">
          <LoadingSpinner size="small" />
        </div>
      )}

      {confirmDeleteModalVisible && (
        <ConfirmDeleteModal
          onConfirm={() => {
            if (selectionMode && selectedIds.size > 0) {
              handleConfirmBulkDelete();
            } else {
              handleConfirmDelete();
            }
            setConfirmDeleteModalVisible(false);
            setSelectedConversationTitle(null);
          }}
          onCancel={() => {
            setConfirmDeleteModalVisible(false);
            setSelectedConversationTitle(null);
          }}
          conversationTitle={
            selectionMode ? undefined : (selectedConversationTitle ?? undefined)
          }
          count={selectionMode ? selectedIds.size : undefined}
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
          onCancel={() => setConfirmExitConversationModalVisible(false)}
        />
      )}
    </div>
  );
}
