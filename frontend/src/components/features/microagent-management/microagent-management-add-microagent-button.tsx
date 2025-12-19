import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useMicroagentManagementStore } from "#/stores/microagent-management-store";
import { GitRepository } from "#/types/git";

interface MicroagentManagementAddMicroagentButtonProps {
  repository: GitRepository;
}

export function MicroagentManagementAddMicroagentButton({
  repository,
}: MicroagentManagementAddMicroagentButtonProps) {
  const { t } = useTranslation();

  const {
    addMicroagentModalVisible,
    setAddMicroagentModalVisible,
    setSelectedRepository,
  } = useMicroagentManagementStore();

  const handleClick = (e: React.MouseEvent<HTMLSpanElement>) => {
    e.stopPropagation();
    e.preventDefault();
    setAddMicroagentModalVisible(!addMicroagentModalVisible);
    setSelectedRepository(repository);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLSpanElement>) => {
    if (e.key === "Enter" || e.key === " ") {
      e.stopPropagation();
      e.preventDefault();
      setAddMicroagentModalVisible(!addMicroagentModalVisible);
      setSelectedRepository(repository);
    }
  };

  return (
    <span
      role="button"
      tabIndex={0}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      className="translate-y-[-1px] text-sm font-normal leading-5 text-[#8480FF] cursor-pointer hover:text-[#6C63FF] transition-colors duration-200"
      data-testid="add-microagent-button"
    >
      {t(I18nKey.COMMON$ADD_MICROAGENT)}
    </span>
  );
}
