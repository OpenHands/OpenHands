import { useTranslation } from "react-i18next";
import { formatTimeDelta } from "#/utils/format-time-delta";
import { cn } from "#/utils/utils";
import { I18nKey } from "#/i18n/declaration";
import { RepositorySelection } from "#/api/open-hands.types";
import { ConversationRepoLink } from "./conversation-repo-link";
import { NoRepository } from "./no-repository";
import { ConversationStatus } from "#/types/conversation-status";
import CircuitIcon from "#/icons/u-circuit.svg?react";

interface ConversationCardFooterProps {
  selectedRepository: RepositorySelection | null;
  lastUpdatedAt: string; // ISO 8601
  createdAt?: string; // ISO 8601
  conversationStatus?: ConversationStatus;
  llmModel?: string | null;
}

export function ConversationCardFooter({
  selectedRepository,
  lastUpdatedAt,
  createdAt,
  conversationStatus,
  llmModel,
}: ConversationCardFooterProps) {
  const { t } = useTranslation();

  const isConversationArchived = conversationStatus === "ARCHIVED";

  return (
    <div
      className={cn(
        "flex flex-row justify-between items-center mt-1",
        isConversationArchived && "opacity-60",
      )}
    >
      {selectedRepository?.selected_repository ? (
        <ConversationRepoLink selectedRepository={selectedRepository} />
      ) : (
        <NoRepository />
      )}
      <div className="flex items-center gap-2 flex-1 justify-end">
        {llmModel && (
          <span
            className="text-xs text-[#A3A3A3] max-w-[120px] flex items-center gap-1 overflow-hidden"
            title={llmModel}
            data-testid="conversation-card-llm-model"
          >
            <CircuitIcon width={12} height={12} className="shrink-0" />
            <span className="truncate">{llmModel}</span>
          </span>
        )}
        {(createdAt ?? lastUpdatedAt) && (
          <p className="text-xs text-[#A3A3A3] text-right">
            <time>
              {`${formatTimeDelta(lastUpdatedAt ?? createdAt)} ${t(I18nKey.CONVERSATION$AGO)}`}
            </time>
          </p>
        )}
      </div>
    </div>
  );
}
