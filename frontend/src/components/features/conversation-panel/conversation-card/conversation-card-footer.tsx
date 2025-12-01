import { useTranslation } from "react-i18next";
import { formatTimeDelta } from "#/utils/format-time-delta";
import { cn } from "#/utils/utils";
import { I18nKey } from "#/i18n/declaration";
import { RepositorySelection } from "#/api/open-hands.types";
import { ConversationRepoLink } from "./conversation-repo-link";
import { NoRepository } from "./no-repository";
import { ConversationStatus } from "#/types/conversation-status";

interface ConversationCardFooterProps {
  selectedRepository: RepositorySelection | null;
  lastUpdatedAt: string; // ISO 8601
  createdAt?: string; // ISO 8601
  conversationStatus?: ConversationStatus;
}

export function ConversationCardFooter({
  selectedRepository,
  lastUpdatedAt,
  createdAt,
  conversationStatus,
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
      {(createdAt ?? lastUpdatedAt) && (
        <p className="text-xs text-[#A3A3A3] flex-1 text-right">
          <time>
            {`${formatTimeDelta(lastUpdatedAt ?? createdAt)} ${t(I18nKey.CONVERSATION$AGO)}`}
          </time>
        </p>
      )}
    </div>
  );
}
