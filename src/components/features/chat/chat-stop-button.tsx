import { useTranslation } from "react-i18next";
import PauseIcon from "#/icons/pause.svg?react";
import { I18nKey } from "#/i18n/declaration";

export interface ChatStopButtonProps {
  handleStop: () => void;
}

export function ChatStopButton({ handleStop }: ChatStopButtonProps) {
  const { t } = useTranslation("openhands");

  return (
    <button
      type="button"
      onClick={handleStop}
      data-testid="stop-button"
      aria-label={t(I18nKey.BUTTON$STOP)}
      className="cursor-pointer"
    >
      <PauseIcon className="block max-w-none w-4 h-4 text-current" />
    </button>
  );
}
