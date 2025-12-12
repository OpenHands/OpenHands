import { useTranslation } from "react-i18next";
import { Accordion, AccordionItem, Spinner } from "@heroui/react";
import { MicroagentManagementRepoMicroagents } from "./microagent-management-repo-microagents";
import { GitRepository } from "#/types/git";
import { TabType } from "#/types/microagent-management";
import { MicroagentManagementNoRepositories } from "./microagent-management-no-repositories";
import { I18nKey } from "#/i18n/declaration";
import { DOCUMENTATION_URL } from "#/utils/constants";
import { MicroagentManagementAccordionTitle } from "./microagent-management-accordion-title";

type MicroagentManagementRepositoriesProps = {
  repositories: GitRepository[];
  tabType: TabType;
  isSearchLoading?: boolean;
};

export function MicroagentManagementRepositories({
  repositories,
  tabType,
  isSearchLoading = false,
}: MicroagentManagementRepositoriesProps) {
  const { t } = useTranslation();

  const numberOfRepoMicroagents = repositories.length;

  // Show spinner when search is in progress, regardless of repository count
  if (isSearchLoading) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 py-8">
        <Spinner size="sm" />
        <span className="text-sm text-white">
          {t("HOME$SEARCHING_REPOSITORIES")}
        </span>
      </div>
    );
  }

  if (numberOfRepoMicroagents === 0) {
    if (tabType === "personal") {
      return (
        <MicroagentManagementNoRepositories
          title={t(
            I18nKey.MICROAGENT_MANAGEMENT$YOU_DO_NOT_HAVE_USER_LEVEL_MICROAGENTS,
          )}
          documentationUrl={DOCUMENTATION_URL.MICROAGENTS.MICROAGENTS_OVERVIEW}
        />
      );
    }
    if (tabType === "repositories") {
      return (
        <MicroagentManagementNoRepositories
          title={t(I18nKey.MICROAGENT_MANAGEMENT$YOU_DO_NOT_HAVE_MICROAGENTS)}
          documentationUrl={DOCUMENTATION_URL.MICROAGENTS.MICROAGENTS_OVERVIEW}
        />
      );
    }
    if (tabType === "organizations") {
      return (
        <MicroagentManagementNoRepositories
          title={t(
            I18nKey.MICROAGENT_MANAGEMENT$YOU_DO_NOT_HAVE_ORGANIZATION_LEVEL_MICROAGENTS,
          )}
          documentationUrl={
            DOCUMENTATION_URL.MICROAGENTS.ORGANIZATION_AND_USER_MICROAGENTS
          }
        />
      );
    }
  }

  return (
    <div className="flex flex-col gap-4 w-full">
      {/* Repositories Accordion */}
      <Accordion
        variant="splitted"
        className="w-full px-0 gap-3"
        itemClasses={{
          base: "shadow-none bg-transparent cursor-pointer px-0",
          trigger: "cursor-pointer gap-2 py-3",
          indicator:
            "flex items-center justify-center p-0.5 pr-[3px] text-white hover:bg-[#454545] rounded transition-colors duration-200 rotate-180",
        }}
        selectionMode="multiple"
      >
        {repositories.map((repository) => (
          <AccordionItem
            key={repository.id}
            aria-label={repository.full_name}
            title={
              <MicroagentManagementAccordionTitle repository={repository} />
            }
          >
            <MicroagentManagementRepoMicroagents repository={repository} />
          </AccordionItem>
        ))}
      </Accordion>
    </div>
  );
}
