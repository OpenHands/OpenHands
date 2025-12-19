import { GitProviderIcon } from "#/components/shared/git-provider-icon";
import { GitRepository } from "#/types/git";
import { MicroagentManagementAddMicroagentButton } from "./microagent-management-add-microagent-button";
import { TooltipButton } from "#/components/shared/buttons/tooltip-button";

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
        <TooltipButton
          tooltip={repository.full_name}
          ariaLabel={repository.full_name}
          testId="repository-name-tooltip"
          placement="bottom"
          asSpan
          className="text-white text-base font-normal bg-transparent p-0 min-w-0 h-auto cursor-pointer truncate max-w-[194px] translate-y-[-1px]"
        >
          <span>{repository.full_name}</span>
        </TooltipButton>
      </div>
      <MicroagentManagementAddMicroagentButton repository={repository} />
    </div>
  );
}
