import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useMicroagentManagementStore } from "#/stores/microagent-management-store";
import { GitRepository } from "#/types/git";

interface MicroagentManagementLearnThisRepoProps {
  repository: GitRepository;
}

export function MicroagentManagementLearnThisRepo({
  repository,
}: MicroagentManagementLearnThisRepoProps) {
  const { setLearnThisRepoModalVisible, setSelectedRepository } =
    useMicroagentManagementStore();
  const { t } = useTranslation();

  const handleClick = () => {
    setLearnThisRepoModalVisible(true);
    setSelectedRepository(repository);
  };

  return (
    <div
      className="flex items-center justify-center rounded-lg bg-white/5 border border-dashed border-white/30 p-4 hover:bg-white/20 hover:border-primary transition-all duration-300 cursor-pointer"
      onClick={handleClick}
      data-testid="learn-this-repo-trigger"
    >
      <span className="text-[16px] font-normal text-[#8480FF]">
        {t(I18nKey.MICROAGENT_MANAGEMENT$LEARN_THIS_REPO)}
      </span>
    </div>
  );
}
