import React from "react";
import { Tooltip } from "@heroui/react";
import { useTranslation } from "react-i18next";
import { NavigationLink } from "#/components/shared/navigation-link";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import { SandboxStatus } from "#/api/conversation-service/agent-server-conversation-service.types";
import { RepositorySelection } from "#/api/open-hands.types";
import { cn } from "#/utils/utils";
import { ConversationStatusDot } from "./conversation-status-dot";
import { ConversationCardFooter } from "./conversation-card/conversation-card-footer";
import { I18nKey } from "#/i18n/declaration";
import { useBackendScopedPath } from "#/hooks/use-backend-scoped-path";
import { InsiderCatBadge } from "../conversation/insider-cat-badge";
import { isInsiderConversation } from "#/utils/insider-cat";

interface CompactConversationRowProps {
  conversationId: string;
  title: string;
  selectedRepository: RepositorySelection | null;
  executionStatus?: ExecutionStatus | null;
  sandboxStatus?: SandboxStatus | null;
  lastUpdatedAt: string;
  createdAt?: string;
  workspaceWorkingDir?: string | null;
  isActive?: boolean;
  onClose?: () => void;
  showRepositoryMetadata?: boolean;
  llmModel?: string | null;
  showLlmProfiles?: boolean;
  agentKind?: "openhands" | "acp" | null;
  acpServer?: string | null;
  tags?: Record<string, string> | null;
  parentConversationId?: string | null;
  showTags?: boolean;
}

/**
 * Minimal one-row presentation of a conversation used by the collapsed
 * sidebar. The row itself is just the agent status dot; hovering it shows a
 * floating preview with the conversation's title, repo and timestamp.
 */
export function CompactConversationRow({
  conversationId,
  title,
  selectedRepository,
  executionStatus,
  sandboxStatus,
  lastUpdatedAt,
  createdAt,
  workspaceWorkingDir,
  isActive = false,
  onClose,
  showRepositoryMetadata = true,
  llmModel = null,
  showLlmProfiles = false,
  agentKind = null,
  acpServer = null,
  tags = null,
  parentConversationId,
  showTags = false,
}: CompactConversationRowProps) {
  const { t } = useTranslation("openhands");
  const backendScopedPath = useBackendScopedPath();
  const disableAnimation = import.meta.env.MODE === "test";

  const preview = (
    <div className="w-65 p-3">
      <div className="flex items-center gap-2 mb-1">
        <InsiderCatBadge
          tags={tags}
          parentConversationId={parentConversationId}
          compact
        />
        <ConversationStatusDot
          executionStatus={executionStatus}
          sandboxStatus={sandboxStatus}
          showTooltip={false}
        />
        <span
          className="text-sm font-medium text-contrast truncate"
          title={title}
        >
          {title || t(I18nKey.CONVERSATION$UNTITLED)}
        </span>
      </div>
      <ConversationCardFooter
        selectedRepository={selectedRepository}
        lastUpdatedAt={lastUpdatedAt}
        createdAt={createdAt}
        executionStatus={executionStatus}
        workspaceWorkingDir={workspaceWorkingDir}
        showRepositoryMetadata={showRepositoryMetadata}
        llmModel={llmModel}
        showAgentChip={showLlmProfiles}
        agentKind={agentKind}
        acpServer={acpServer}
        tags={tags}
        showTags={showTags}
        parentConversationId={parentConversationId}
      />
    </div>
  );

  return (
    <Tooltip
      content={preview}
      placement="right"
      closeDelay={100}
      className="bg-surface text-contrast border border-border-subtle shadow-xl p-0"
      disableAnimation={disableAnimation}
    >
      <NavigationLink
        to={backendScopedPath(`/conversations/${conversationId}`)}
        onClick={onClose}
        data-testid="compact-conversation-row"
        data-conversation-id={conversationId}
        aria-label={title || conversationId}
        className={({ isActive: navActive }) =>
          cn(
            "flex items-center justify-center w-10 h-9 mx-auto rounded-md",
            "transition-colors cursor-pointer",
            navActive || isActive ? "bg-tertiary" : "hover:bg-surface-raised",
          )
        }
      >
        <span className="relative inline-flex items-center justify-center">
          <InsiderCatBadge
            tags={tags}
            parentConversationId={parentConversationId}
            compact
          />
          <span
            className={
              isInsiderConversation({
                tags,
                parent_conversation_id: parentConversationId,
              })
                ? "absolute -bottom-1 -right-1 rounded-full bg-[var(--oh-surface)] p-0.5"
                : undefined
            }
          >
            <ConversationStatusDot
              executionStatus={executionStatus}
              sandboxStatus={sandboxStatus}
              showTooltip={false}
            />
          </span>
        </span>
      </NavigationLink>
    </Tooltip>
  );
}
