import { useTranslation } from "react-i18next";
import { Link } from "react-router";
import CodeBranchIcon from "#/icons/u-code-branch.svg?react";
import { Conversation } from "#/api/open-hands.types";
import { GitProviderIcon } from "#/components/shared/git-provider-icon";
import { Provider } from "#/types/settings";
import { formatTimeDelta } from "#/utils/format-time-delta";
import { I18nKey } from "#/i18n/declaration";
import { ConversationStatusIndicator } from "./conversation-status-indicator";
import RepoForkedIcon from "#/icons/repo-forked.svg?react";

interface RecentConversationProps {
  conversation: Conversation;
}

export function RecentConversation({ conversation }: RecentConversationProps) {
  const { t } = useTranslation();

  const hasRepository =
    conversation.selected_repository && conversation.selected_branch;

  return (
    <Link
      to={`/conversations/${conversation.conversation_id}`}
      className="flex flex-col gap-1 p-[14px] cursor-pointer w-full rounded-lg hover:bg-[#5C5D62] transition-all duration-300 text-left"
    >
      <div className="flex items-center gap-2 pl-1">
        <ConversationStatusIndicator conversationStatus={conversation.status} />
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
        {(conversation.created_at || conversation.last_updated_at) && (
          <span>
            {formatTimeDelta(
              conversation.created_at || conversation.last_updated_at,
            )}{" "}
            {t(I18nKey.CONVERSATION$AGO)}
          </span>
        )}
      </div>
    </Link>
  );
}
