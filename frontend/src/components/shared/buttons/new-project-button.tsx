import { useNavigate } from "react-router";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { useIsCreatingConversation } from "#/hooks/use-is-creating-conversation";
import PlusIcon from "#/icons/u-plus.svg?react";
import { cn } from "#/utils/utils";

interface NewProjectButtonProps {
  disabled?: boolean;
}

export function NewProjectButton({ disabled = false }: NewProjectButtonProps) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const {
    mutate: createConversation,
    isPending,
    isSuccess,
  } = useCreateConversation();
  const isCreatingConversationElsewhere = useIsCreatingConversation();

  const startNewProject = t(I18nKey.CONVERSATION$START_NEW);
  const isCreatingConversation =
    isPending || isSuccess || isCreatingConversationElsewhere;
  const isDisabled = disabled || isCreatingConversation;

  const handleCreateConversation = () => {
    if (isDisabled) {
      return;
    }

    createConversation(
      {},
      {
        onSuccess: (data) => navigate(`/conversations/${data.conversation_id}`),
      },
    );
  };

  return (
    <StyledTooltip content={startNewProject} placement="right">
      <button
        type="button"
        data-testid="new-project-button"
        aria-label={startNewProject}
        tabIndex={isDisabled ? -1 : 0}
        onClick={handleCreateConversation}
        disabled={isDisabled}
        className={cn("inline-flex items-center justify-center", {
          "pointer-events-none opacity-50": isDisabled,
        })}
      >
        <PlusIcon width={24} height={24} />
      </button>
    </StyledTooltip>
  );
}
