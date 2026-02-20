import { useTranslation } from "react-i18next";
import { formatTimeDelta } from "#/utils/format-time-delta";
import { cn } from "#/utils/utils";
import { I18nKey } from "#/i18n/declaration";
import {
  ConversationTrigger,
  RepositorySelection,
} from "#/api/open-hands.types";
import { ConversationRepoLink } from "./conversation-repo-link";
import { NoRepository } from "./no-repository";
import { EnvironmentLink } from "./environment-link";
import { ConversationStatus } from "#/types/conversation-status";

interface ConversationCardFooterProps {
  selectedRepository: RepositorySelection | null;
  lastUpdatedAt: string; // ISO 8601
  createdAt?: string; // ISO 8601
  conversationStatus?: ConversationStatus;
  trigger?: ConversationTrigger;
  environmentUrl?: string | null;
}

export function ConversationCardFooter({
  selectedRepository,
  lastUpdatedAt,
  createdAt,
  conversationStatus,
  trigger,
  environmentUrl,
}: ConversationCardFooterProps) {
  const { t } = useTranslation();

  const isConversationArchived = conversationStatus === "ARCHIVED";

  const renderSource = () => {
    if (selectedRepository?.selected_repository) {
      return <ConversationRepoLink selectedRepository={selectedRepository} />;
    }
    if (trigger === "connect_to_environment") {
      const url = environmentUrl || null;
      if (url) return <EnvironmentLink url={url} />;
    }
    return <NoRepository />;
  };

  return (
    <div
      className={cn(
        "flex flex-row justify-between items-center mt-1",
        isConversationArchived && "opacity-60",
      )}
    >
      {renderSource()}
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
