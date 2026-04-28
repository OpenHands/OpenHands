import { useTranslation } from "react-i18next";
import { RefreshCw } from "lucide-react";
import { BaseModalTitle } from "#/components/shared/modals/confirmation-modals/base-modal";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "../settings/brand-button";

interface SkillsModalHeaderProps {
  isAgentReady: boolean;
  isLoading: boolean;
  isRefetching: boolean;
  skillCount: number;
  allExpanded: boolean;
  onRefresh: () => void;
  onToggleAll: () => void;
}

export function SkillsModalHeader({
  isAgentReady,
  isLoading,
  isRefetching,
  skillCount,
  allExpanded,
  onRefresh,
  onToggleAll,
}: SkillsModalHeaderProps) {
  const { t } = useTranslation();

  return (
    <div className="flex flex-col gap-6 w-full">
      <div className="flex items-center justify-between w-full">
        <BaseModalTitle title={`${t(I18nKey.SKILLS_MODAL$TITLE)} (${skillCount})`} />
        {isAgentReady && (
          <div className="flex items-center gap-2">
            <BrandButton
              testId="toggle-all-skills"
              type="button"
              variant="secondary"
              onClick={onToggleAll}
              isDisabled={isLoading || isRefetching || skillCount === 0}
            >
              {allExpanded
                ? t(I18nKey.SKILLS_MODAL$COLLAPSE_ALL)
                : t(I18nKey.SKILLS_MODAL$EXPAND_ALL)}
            </BrandButton>
            <BrandButton
              testId="refresh-skills"
              type="button"
              variant="primary"
              className="flex items-center gap-2"
              onClick={onRefresh}
              isDisabled={isLoading || isRefetching}
            >
              <RefreshCw
                size={16}
                className={`${isRefetching ? "animate-spin" : ""}`}
              />
              {t(I18nKey.BUTTON$REFRESH)}
            </BrandButton>
          </div>
        )}
      </div>
    </div>
  );
}
