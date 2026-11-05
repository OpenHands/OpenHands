import { useTranslation } from "react-i18next";
import { X } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";

interface ClearButtonProps {
  onClear: () => void;
}

export function ClearButton({ onClear }: ClearButtonProps) {
  const { t } = useTranslation("openhands");
  return (
    <button
      type="button"
      data-testid="dropdown-clear"
      onClick={onClear}
      aria-label={t(I18nKey.COMMON$CLEAR_SELECTION)}
      className="text-contrast hover:text-text-tertiary"
    >
      <X size={14} />
    </button>
  );
}
