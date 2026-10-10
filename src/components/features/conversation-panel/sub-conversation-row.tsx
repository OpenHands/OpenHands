import React from "react";
import { Tooltip } from "@heroui/react";
import { useTranslation } from "react-i18next";
import { NavigationLink } from "#/components/shared/navigation-link";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import { SandboxStatus } from "#/api/conversation-service/agent-server-conversation-service.types";
import { I18nKey } from "#/i18n/declaration";
import { useBackendScopedPath } from "#/hooks/use-backend-scoped-path";
import { cn } from "#/utils/utils";
import { formatTimeDelta } from "#/utils/format-time-delta";
import { ConversationStatusDot } from "./conversation-status-dot";

interface SubConversationRowProps {
  conversationId: string;
  title: string;
  executionStatus?: ExecutionStatus | null;
  sandboxStatus?: SandboxStatus | null;
  lastUpdatedAt: string;
  isActive?: boolean;
  onClose?: () => void;
}

/**
 * Indented child row rendered under a parent conversation card when its
 * sub-conversations are expanded. Deliberately minimal — the parent card
 * already carries the rich metadata (repo, model, tags); the child only
 * needs to be identifiable and clickable.
 */
export function SubConversationRow({
  conversationId,
  title,
  executionStatus,
  sandboxStatus,
  lastUpdatedAt,
  isActive = false,
  onClose,
}: SubConversationRowProps) {
  const { t } = useTranslation("openhands");
  const backendScopedPath = useBackendScopedPath();
  const disableAnimation = import.meta.env.MODE === "test";

  return (
    <Tooltip
      content={title || t(I18nKey.CONVERSATION$UNTITLED)}
      placement="right"
      delay={1000}
      closeDelay={100}
      isDisabled={import.meta.env.MODE === "test"}
      className="border border-border bg-base-secondary text-contrast rounded-md px-2 py-1 max-w-65"
      disableAnimation={disableAnimation}
    >
      <NavigationLink
        to={backendScopedPath(`/conversations/${conversationId}`)}
        onClick={onClose}
        data-testid="sub-conversation-row"
        data-conversation-id={conversationId}
        aria-label={title || conversationId}
        className={cn(
          "flex w-full min-w-0 items-center gap-2 rounded-md py-1 pl-2 pr-1",
          "transition-colors cursor-pointer",
          isActive ? "bg-surface" : "hover:bg-surface",
        )}
      >
        <ConversationStatusDot
          executionStatus={executionStatus}
          sandboxStatus={sandboxStatus}
          showTooltip={false}
        />
        <span
          className="min-w-0 flex-1 truncate text-xs text-contrast"
          title={title || conversationId}
        >
          {title || t(I18nKey.CONVERSATION$UNTITLED)}
        </span>
        <span className="shrink-0 pr-1 text-xs text-muted tabular-nums">
          <time>{formatTimeDelta(lastUpdatedAt)}</time>
        </span>
      </NavigationLink>
    </Tooltip>
  );
}
