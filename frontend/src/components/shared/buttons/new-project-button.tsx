import { NavLink } from "react-router";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import PlusIcon from "#/icons/u-plus.svg?react";
import { cn } from "#/utils/utils";

interface NewProjectButtonProps {
  disabled?: boolean;
}

export function NewProjectButton({ disabled = false }: NewProjectButtonProps) {
  const { t } = useTranslation();

  const startNewProject = t(I18nKey.CONVERSATION$START_NEW);

  return (
    <StyledTooltip content={startNewProject} placement="right">
      <NavLink
        to="/"
        data-testid="new-project-button"
        aria-label={startNewProject}
        tabIndex={disabled ? -1 : 0}
        onClick={(e) => {
          if (disabled) {
            e.preventDefault();
          }
        }}
        className={cn("inline-flex items-center justify-center", {
          "pointer-events-none opacity-50": disabled,
        })}
      >
        <PlusIcon width={24} height={24} />
      </NavLink>
    </StyledTooltip>
  );
}
