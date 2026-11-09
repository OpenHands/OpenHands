import { Loader2, Mic, Square } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useDictation, type DictationStatus } from "#/hooks/chat/use-dictation";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { chatInputIconButtonClassName } from "#/utils/form-control-classes";
import { cn } from "#/utils/utils";

const STATUS_ICONS: Record<DictationStatus, typeof Mic> = {
  idle: Mic,
  starting: Loader2,
  recording: Square,
  transcribing: Loader2,
};

export interface ChatDictationButtonProps {
  onTranscript: (text: string) => void;
  disabled?: boolean;
}

export function ChatDictationButton({
  onTranscript,
  disabled = false,
}: ChatDictationButtonProps) {
  const { t } = useTranslation("openhands");
  const { status, isSupported, toggle } = useDictation({
    onTranscript,
    onError: () =>
      displayErrorToast(t(I18nKey.CHAT_INTERFACE$DICTATION_FAILED)),
  });

  if (!isSupported) return null;

  const isRecording = status === "recording";
  const isBusy = status === "starting" || status === "transcribing";
  // Stopping stays possible even if the text field turned read-only.
  const isDisabled = !isRecording && (disabled || status !== "idle");
  const Icon = STATUS_ICONS[status];

  return (
    <button
      type="button"
      className={cn(
        chatInputIconButtonClassName,
        "shrink-0 size-8",
        isRecording && "text-danger",
        isDisabled && "cursor-not-allowed text-text-subtle",
      )}
      aria-label={t(
        isRecording
          ? I18nKey.CHAT_INTERFACE$STOP_DICTATION
          : I18nKey.CHAT_INTERFACE$START_DICTATION,
      )}
      aria-pressed={isRecording}
      aria-busy={isBusy}
      data-testid="chat-dictation-button"
      onClick={toggle}
      disabled={isDisabled}
    >
      <Icon
        className={cn("size-4", isBusy && "animate-spin")}
        fill={isRecording ? "currentColor" : "none"}
        aria-hidden
      />
    </button>
  );
}
