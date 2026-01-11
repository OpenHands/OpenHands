import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import TachometerFastIcon from "#/icons/tachometer-fast.svg?react";
import PrStatusIcon from "#/icons/pr-status.svg?react";
import DocumentIcon from "#/icons/document.svg?react";
import WaterIcon from "#/icons/u-water.svg?react";

export type Suggestion = { label: I18nKey | string; value: string };

interface SuggestionItemProps {
  suggestion: Suggestion;
  onClick: (value: string) => void;
}

export function SuggestionItem({ suggestion, onClick }: SuggestionItemProps) {
  const { t } = useTranslation();

  const itemIcon = useMemo(() => {
    switch (suggestion.label) {
      case "INCREASE_TEST_COVERAGE":
        return (
          <TachometerFastIcon width={24} height={24} className="text-content" />
        );
      case "AUTO_MERGE_PRS":
        return <PrStatusIcon width={19} height={20} className="text-content" />;
      case "FIX_README":
        return <DocumentIcon width={24} height={24} className="text-content" />;
      case "CLEAN_DEPENDENCIES":
        return <WaterIcon width={24} height={24} className="text-content" />;
      default:
        return null;
    }
  }, [suggestion]);

  return (
    <button
      type="button"
      className="list-none border border-stroke-alt rounded-[15px] hover:bg-tertiary flex-1 flex items-center justify-center cursor-pointer gap-[10px] h-[55px] px-4"
      onClick={() => onClick(suggestion.value)}
    >
      {itemIcon}
      <span
        data-testid="suggestion"
        className="text-[15px] font-normal leading-5 text-subtle text-center cursor-pointer"
      >
        {t(suggestion.label)}
      </span>
    </button>
  );
}
