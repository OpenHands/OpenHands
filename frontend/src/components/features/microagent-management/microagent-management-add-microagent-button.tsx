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

  const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    setAddMicroagentModalVisible(!addMicroagentModalVisible);
    setSelectedRepository(repository);
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      className="translate-y-[-1px]"
      data-testid="add-microagent-button"
    >
      <span className="text-sm font-normal leading-5 text-[#8480FF] cursor-pointer hover:text-[#6C63FF] transition-colors duration-200">
        {t(I18nKey.COMMON$ADD_MICROAGENT)}
      </span>
    </button>
  );
}
