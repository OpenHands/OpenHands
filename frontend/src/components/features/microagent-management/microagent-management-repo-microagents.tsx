import { useTranslation } from "react-i18next";
import { useEffect } from "react";
import { Spinner } from "@heroui/react";
import { MicroagentManagementMicroagentCard } from "./microagent-management-microagent-card";
import { MicroagentManagementLearnThisRepo } from "./microagent-management-learn-this-repo";
import { useRepositoryMicroagents } from "#/hooks/query/use-repository-microagents";
import { useMicroagentManagementConversations } from "#/hooks/query/use-microagent-management-conversations";
import { GitRepository } from "#/types/git";
import { useMicroagentManagementStore } from "#/stores/microagent-management-store";
import { cn } from "#/utils/utils";
import { I18nKey } from "#/i18n/declaration";

interface MicroagentManagementRepoMicroagentsProps {
  repository: GitRepository;
}

export function MicroagentManagementRepoMicroagents({
  repository,
}: MicroagentManagementRepoMicroagentsProps) {
  const { selectedMicroagentItem, setSelectedMicroagentItem } =
    useMicroagentManagementStore();

  const { t } = useTranslation();

  const { full_name: repositoryName } = repository;

  // Extract owner and repo from repositoryName (format: "owner/repo")
  const [owner, repo] = repositoryName.split("/");

  const {
    data: microagents,
    isLoading: isLoadingMicroagents,
    isError: isErrorMicroagents,
  } = useRepositoryMicroagents(owner, repo, true);

  const {
    data: conversations,
    isLoading: isLoadingConversations,
    isError: isErrorConversations,
  } = useMicroagentManagementConversations(
    repositoryName,
    undefined,
    1000,
    true,
  );

  useEffect(() => {
    const hasConversations = conversations && conversations.length > 0;
    const selectedConversation = selectedMicroagentItem?.conversation;

    if (hasConversations && selectedConversation) {
      // get the latest selected conversation.
      const latestSelectedConversation = conversations.find(
        (conversation) =>
          conversation.conversation_id === selectedConversation.conversation_id,
      );
      if (latestSelectedConversation) {
        setSelectedMicroagentItem({
          microagent: undefined,
          conversation: latestSelectedConversation,
        });
      }
    }
  }, [conversations]);

  useEffect(
    () => () => {
      setSelectedMicroagentItem({
        microagent: undefined,
        conversation: undefined,
      });
    },
    [setSelectedMicroagentItem],
  );

  // Show loading only when both queries are loading
  const isLoading = isLoadingMicroagents || isLoadingConversations;

  // Show error UI.
  const isError = isErrorMicroagents || isErrorConversations;

  if (isLoading) {
    return (
      <div className="pb-4 flex justify-center">
        <Spinner size="sm" data-testid="loading-spinner" />
      </div>
    );
  }

  // If there's an error with microagents, show the learn this repo component
  if (isError) {
    return (
      <div>
        <MicroagentManagementLearnThisRepo repository={repository} />
      </div>
    );
  }

  const numberOfMicroagents = microagents?.length || 0;
  const numberOfConversations = conversations?.length || 0;
  const totalItems = numberOfMicroagents + numberOfConversations;
  const hasMicroagents = numberOfMicroagents > 0;
  const hasConversations = numberOfConversations > 0;

  return (
    <div>
      {totalItems === 0 && (
        <MicroagentManagementLearnThisRepo repository={repository} />
      )}
      {/* Render microagents */}
      {hasMicroagents && (
        <div className="flex flex-col">
          <span className="text-md text-white font-medium leading-5 mb-4">
            {t(I18nKey.MICROAGENT_MANAGEMENT$EXISTING_MICROAGENTS)}
          </span>
          {microagents?.map((microagent) => (
            <div key={microagent.name} className="pb-4 last:pb-0">
              <MicroagentManagementMicroagentCard
                microagent={microagent}
                repository={repository}
              />
            </div>
          ))}
        </div>
      )}

      {/* Render conversations */}
      {hasConversations && (
        <div className={cn("flex flex-col", hasMicroagents && "mt-4")}>
          <span className="text-md text-white font-medium leading-5 mb-4">
            {t(I18nKey.COMMON$IN_PROGRESS)}
          </span>
          {conversations?.map((conversation) => (
            <div key={conversation.conversation_id} className="pb-4 last:pb-0">
              <MicroagentManagementMicroagentCard
                conversation={conversation}
                repository={repository}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
