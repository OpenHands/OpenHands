import { useTranslation } from "react-i18next";
import { Link } from "react-router";
import CodeBranchIcon from "#/icons/u-code-branch.svg?react";
import { V1AppConversation } from "#/api/conversation-service/v1-conversation-service.types";
import { GitProviderIcon } from "#/components/shared/git-provider-icon";
import { useConfig } from "#/hooks/query/use-config";
import { Provider } from "#/types/settings";
import { formatTimeDelta } from "#/utils/format-time-delta";
import { I18nKey } from "#/i18n/declaration";
import { SandboxStatusIndicator } from "./sandbox-status-indicator";
import RepoForkedIcon from "#/icons/repo-forked.svg?react";
import { Typography } from "#/ui/typography";
import { resolveAgentChip } from "#/utils/agent-display-label";
import { AgentChipIcon } from "#/components/shared/agent-chip-icon";

interface RecentConversationProps {
  conversation: V1AppConversation;
}

export function RecentConversation({ conversation }: RecentConversationProps) {
  const { t } = useTranslation();
  const { data: config } = useConfig();

  const hasRepository =
    conversation.selected_repository && conversation.selected_branch;
  const agentChip = resolveAgentChip(
    conversation.agent_kind,
    conversation.llm_model,
    conversation.acp_server,
    config?.acp_providers,
  );

  return (
    <Link
      to={`/conversations/${conversation.id}`}
      className="flex flex-col gap-1 p-[14px] cursor-pointer w-full rounded-lg hover:bg-[#5C5D62] transition-all duration-300 text-left"
    >
      <div className="flex items-center gap-2 pl-1">
        <SandboxStatusIndicator sandboxStatus={conversation.sandbox_status} />
        <span className="text-xs text-white leading-6 font-normal">
          {conversation.title}
        </span>
      </div>
      <div className="flex items-center justify-between text-xs text-[#A3A3A3] leading-4 font-normal">
        <div className="flex items-center gap-3">
          {hasRepository ? (
            <div className="flex items-center gap-2">
              <GitProviderIcon
                gitProvider={conversation.git_provider as Provider}
              />
              <span
                className="max-w-[124px] truncate"
                title={conversation.selected_repository || ""}
              >
                {conversation.selected_repository}
              </span>
            </div>
          ) : (
            <div className="flex items-center gap-1">
              <RepoForkedIcon width={12} height={12} color="#A3A3A3" />
              <span className="max-w-[124px] truncate">
                {t(I18nKey.COMMON$NO_REPOSITORY)}
              </span>
            </div>
          )}
          {hasRepository ? (
            <div className="flex items-center gap-1">
              <CodeBranchIcon width={12} height={12} color="#A3A3A3" />
              <span
                className="max-w-[124px] truncate"
                title={conversation.selected_branch || ""}
              >
                {conversation.selected_branch}
              </span>
            </div>
          ) : null}
        </div>
        <div className="flex items-center gap-2">
          {agentChip && (
            <span
              className="max-w-[120px] flex items-center gap-1 overflow-hidden"
              title={agentChip.tooltip}
              data-testid="recent-conversation-llm-model"
            >
              <AgentChipIcon kind={agentChip.kind} />
              <Typography.Text className="text-xs truncate">
                {agentChip.text}
              </Typography.Text>
            </span>
          )}
          {(conversation.created_at || conversation.updated_at) && (
            <span>
              {formatTimeDelta(
                conversation.created_at || conversation.updated_at,
              )}{" "}
              {t(I18nKey.CONVERSATION$AGO)}
            </span>
          )}
        </div>
      </div>
    </Link>
  );
}
