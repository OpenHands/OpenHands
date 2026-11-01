import React from "react";
import { Tooltip } from "@heroui/react";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import type { Provider } from "#/types/settings";
import { NavigationLink } from "#/components/shared/navigation-link";
import { cn } from "#/utils/utils";
import {
  isMissingSandboxStatus,
  isRuntimeUnavailableSandboxStatus,
} from "#/utils/conversation-archive-status";
import { ConversationCard } from "./conversation-card/conversation-card";
import { ConversationCardPreview } from "./conversation-card/conversation-card-preview";

interface ConversationPanelRowProps {
  conversation: AppConversation;
  displayTitle: string;
  timestampLabel: string;
  href: string;
  isActive: boolean;
  isPinned: boolean;
  isArchived: boolean;
  explicitlyArchived: boolean;
  contextMenuOpen: boolean;
  inPinnedSection: boolean;
  allowEditTags: boolean;
  showHoverMetadata: boolean;
  showRepoBranchMetadata: boolean;
  showLlmProfiles: boolean;
  showTagsMetadata: boolean;
  agentProfileName: string | null | undefined;
  onClose?: () => void;
  handleDeleteProject: (id: string, title: string) => void;
  handleArchiveProject: (id: string, title: string) => void;
  handleUnarchiveProject: (id: string) => void;
  handleStopConversation: (id: string) => void;
  handleNewConversationInSandbox: (sandboxId: string) => void;
  handleEditTags: (id: string) => void;
  handleConversationTitleChange: (id: string, title: string) => void;
  handleTogglePin: (conversation: AppConversation) => void;
  setOpenContextMenuId: (id: string | null) => void;
}

export const ConversationPanelRow = React.memo(function ConversationPanelRow({
  conversation,
  displayTitle,
  timestampLabel,
  href,
  isActive,
  isPinned,
  isArchived,
  explicitlyArchived,
  contextMenuOpen,
  inPinnedSection,
  allowEditTags,
  showHoverMetadata,
  showRepoBranchMetadata,
  showLlmProfiles,
  showTagsMetadata,
  agentProfileName,
  onClose,
  handleDeleteProject,
  handleArchiveProject,
  handleUnarchiveProject,
  handleStopConversation,
  handleNewConversationInSandbox,
  handleEditTags,
  handleConversationTitleChange,
  handleTogglePin,
  setOpenContextMenuId,
}: ConversationPanelRowProps) {
  const runtimeMissing = isMissingSandboxStatus(conversation.sandbox_status);
  const sandboxId = conversation.sandbox_id;
  return (
    <Tooltip
      placement="right-start"
      // Clear the sidebar's resize handle so the preview never covers it.
      offset={20}
      delay={1000}
      closeDelay={100}
      isDisabled={!showHoverMetadata || contextMenuOpen}
      disableAnimation={import.meta.env.MODE === "test"}
      className="max-w-none overflow-visible rounded-xl border border-border bg-base-secondary p-0 text-contrast shadow-xl"
      content={
        <ConversationCardPreview
          title={displayTitle}
          executionStatus={conversation.execution_status}
          sandboxStatus={conversation.sandbox_status}
          selectedRepository={{
            selected_repository: conversation.selected_repository,
            selected_branch: conversation.selected_branch,
            git_provider: conversation.git_provider as Provider,
          }}
          workspaceWorkingDir={
            conversation.selected_workspace ??
            conversation.workspace?.working_dir
          }
          llmModel={conversation.llm_model}
          agentKind={conversation.agent_kind}
          acpServer={conversation.acp_server}
          createdAt={conversation.created_at}
          agentProfileId={
            conversation.launched_agent_profile?.agent_profile_id ?? null
          }
          agentProfileName={agentProfileName}
          tags={conversation.tags}
          isArchived={isArchived}
        />
      }
    >
      <NavigationLink
        to={href}
        onClick={onClose}
        className={cn(
          "block rounded-md transition-colors",
          !contextMenuOpen && "hover:bg-surface",
          (isActive || contextMenuOpen) && "bg-surface",
        )}
      >
        <ConversationCard
          onDelete={() => handleDeleteProject(conversation.id, displayTitle)}
          // Archive is offered until the row is effectively archived; the
          // reverse direction, "Unarchive", is offered only when the row is
          // explicitly archived and its runtime is not missing. A
          // missing-runtime row offers neither.
          onArchive={
            isArchived
              ? undefined
              : () => handleArchiveProject(conversation.id, displayTitle)
          }
          onUnarchive={
            explicitlyArchived && !runtimeMissing
              ? () => handleUnarchiveProject(conversation.id)
              : undefined
          }
          onStop={() => handleStopConversation(conversation.id)}
          onNewConversationInSandbox={
            sandboxId &&
            !isRuntimeUnavailableSandboxStatus(conversation.sandbox_status)
              ? () => handleNewConversationInSandbox(sandboxId)
              : undefined
          }
          onEditTags={
            allowEditTags ? () => handleEditTags(conversation.id) : undefined
          }
          onChangeTitle={(title) =>
            handleConversationTitleChange(conversation.id, title)
          }
          title={displayTitle}
          selectedRepository={{
            selected_repository: conversation.selected_repository,
            selected_branch: conversation.selected_branch,
            git_provider: conversation.git_provider as Provider,
          }}
          timestampLabel={timestampLabel}
          lastUpdatedAt={conversation.updated_at}
          createdAt={conversation.created_at}
          executionStatus={conversation.execution_status}
          sandboxStatus={conversation.sandbox_status}
          conversationId={conversation.id}
          contextMenuOpen={contextMenuOpen}
          onContextMenuToggle={(isOpen) =>
            setOpenContextMenuId(isOpen ? conversation.id : null)
          }
          isActive={isActive}
          workspaceWorkingDir={
            conversation.selected_workspace ??
            conversation.workspace?.working_dir
          }
          showRepositoryMetadata={showRepoBranchMetadata}
          llmModel={conversation.llm_model}
          showLlmProfiles={showLlmProfiles}
          agentKind={conversation.agent_kind}
          acpServer={conversation.acp_server}
          acpModels={conversation.acp_available_models}
          tags={conversation.tags}
          showTags={showTagsMetadata}
          isArchived={isArchived}
          isPinned={isPinned}
          onTogglePin={() => handleTogglePin(conversation)}
          alwaysShowPinIcon={isPinned && !inPinnedSection}
        />
      </NavigationLink>
    </Tooltip>
  );
});
