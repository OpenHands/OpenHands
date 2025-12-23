import { useTranslation } from "react-i18next";
import { Spinner } from "@heroui/react";
import { useMicroagentManagementStore } from "#/stores/microagent-management-store";
import { useRepositoryMicroagentContent } from "#/hooks/query/use-repository-microagent-content";
import { I18nKey } from "#/i18n/declaration";
import { extractRepositoryInfo } from "#/utils/utils";
import { MarkdownRenderer } from "../markdown/markdown-renderer";

export function MicroagentManagementViewMicroagentContent() {
  const { t } = useTranslation();
  const { selectedMicroagentItem, selectedRepository } =
    useMicroagentManagementStore();

  const { microagent } = selectedMicroagentItem ?? {};

  // Extract owner and repo from full_name (e.g., "owner/repo")
  const { owner, repo, filePath } = extractRepositoryInfo(
    selectedRepository,
    microagent,
  );

  // Fetch microagent content using the new API
  const {
    data: microagentData,
    isLoading,
    error,
  } = useRepositoryMicroagentContent(owner, repo, filePath, true);

  if (!microagent || !selectedRepository) {
    return null;
  }

  return (
    <div className="w-full h-full p-6 bg-[#ffffff1a] rounded-2xl text-white text-sm">
      {isLoading && (
        <div className="flex items-center justify-center w-full h-full">
          <Spinner size="lg" data-testid="loading-microagent-content-spinner" />
        </div>
      )}
      {error && (
        <div className="flex items-center justify-center w-full h-full">
          {t(I18nKey.MICROAGENT_MANAGEMENT$ERROR_LOADING_MICROAGENT_CONTENT)}
        </div>
      )}
      {microagentData && !isLoading && !error && (
        <MarkdownRenderer includeStandard>
          {microagentData.content}
        </MarkdownRenderer>
      )}
    </div>
  );
}
