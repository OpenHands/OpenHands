import { useTranslation } from "react-i18next";
import {
  ComboboxCaretIcon,
  comboboxCaretButtonClassName,
} from "#/ui/combobox-caret";
import { cn } from "#/utils/utils";
import { I18nKey } from "#/i18n/declaration";

interface ToggleButtonProps {
  isOpen: boolean;
  disabled: boolean;
  getToggleButtonProps: (
    props?: Record<string, unknown>,
  ) => Record<string, unknown>;
  iconClassName?: string;
}

export function ToggleButton({
  isOpen,
  disabled,
  getToggleButtonProps,
  iconClassName,
}: ToggleButtonProps) {
  const { t } = useTranslation("openhands");

  return (
    <button
      {...getToggleButtonProps({
        disabled,
        className: cn(
          comboboxCaretButtonClassName,
          "text-current",
          isOpen && "rotate-180",
          disabled && "cursor-not-allowed opacity-60",
        ),
      })}
      type="button"
      aria-label={t(I18nKey.COMMON$TOGGLE_MENU)}
    >
      <ComboboxCaretIcon className={iconClassName} />
    </button>
  );
}
