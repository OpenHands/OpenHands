import { GitProviderIcon } from "#/components/shared/git-provider-icon";
import { GitRepository } from "#/types/git";
import { MicroagentManagementAddMicroagentButton } from "./microagent-management-add-microagent-button";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";

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

        <StyledTooltip content={repository.full_name} placement="bottom">
          <span
            className="text-white text-base font-normal bg-transparent p-0 min-w-0 h-auto cursor-pointer truncate max-w-[194px] translate-y-[-1px]"
            data-testid="repository-name-tooltip"
          >
            {repository.full_name}
          </span>
        </StyledTooltip>
      </div>

      <MicroagentManagementAddMicroagentButton repository={repository} />
    </div>
  );
}
