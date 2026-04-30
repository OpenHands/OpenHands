import { useTranslation } from "react-i18next";
import { RefreshCw } from "lucide-react";
import { BaseModalTitle } from "#/components/shared/modals/confirmation-modals/base-modal";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "../settings/brand-button";

interface HooksModalHeaderProps {
  isAgentReady: boolean;
  isLoading: boolean;
  isRefetching: boolean;
  hookCount: number;
  allExpanded: boolean;
  onRefresh: () => void;
  onToggleAll: () => void;
}

export function HooksModalHeader({
  isAgentReady,
  isLoading,
  isRefetching,
  hookCount,
  allExpanded,
  onRefresh,
  onToggleAll,
}: HooksModalHeaderProps) {
  const { t } = useTranslation();

  return (
    <div className="flex flex-col gap-6 w-full">
      <div className="flex items-center justify-between w-full">
        <BaseModalTitle
          title={`${t(I18nKey.HOOKS_MODAL$TITLE)} (${hookCount})`}
        />
        {isAgentReady && (
          <div className="flex items-center gap-2">
            <BrandButton
              testId="toggle-all-hooks"
              type="button"
              variant="secondary"
              onClick={onToggleAll}
              isDisabled={isLoading || isRefetching || hookCount === 0}
            >
              {allExpanded
                ? t(I18nKey.HOOKS_MODAL$COLLAPSE_ALL)
                : t(I18nKey.HOOKS_MODAL$EXPAND_ALL)}
            </BrandButton>
            <BrandButton
              testId="refresh-hooks"
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
