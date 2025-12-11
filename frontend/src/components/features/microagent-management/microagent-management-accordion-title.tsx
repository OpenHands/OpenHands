import { Tooltip } from "@heroui/react";
import { GitProviderIcon } from "#/components/shared/git-provider-icon";
import { GitRepository } from "#/types/git";
import { MicroagentManagementAddMicroagentButton } from "./microagent-management-add-microagent-button";

interface MicroagentManagementAccordionTitleProps {
  repository: GitRepository;
}

export function MicroagentManagementAccordionTitle({
  repository,
}: MicroagentManagementAccordionTitleProps) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2">
        <GitProviderIcon gitProvider={repository.git_provider} />
        <Tooltip
          content={repository.full_name}
          closeDelay={100}
          placement="bottom"
        >
          <span
            data-testid="repository-name-tooltip"
            className="text-white text-base font-normal bg-transparent p-0 min-w-0 h-auto cursor-pointer truncate max-w-[194px] translate-y-[-1px]"
          >
            {repository.full_name}
          </span>
        </Tooltip>
      </div>
      <MicroagentManagementAddMicroagentButton repository={repository} />
    </div>
  );
}
