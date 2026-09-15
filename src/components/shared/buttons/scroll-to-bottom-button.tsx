import ArrowSendIcon from "#/icons/arrow-send.svg?react";

interface ScrollToBottomButtonProps {
  onClick: () => void;
}

export function ScrollToBottomButton({ onClick }: ScrollToBottomButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      data-testid="scroll-to-bottom"
      className="flex items-center justify-center size-8 rounded-full bg-surface text-muted hover:bg-tertiary hover:text-white rotate-180 cursor-pointer transition-colors"
    >
      <ArrowSendIcon width={15} height={15} />
    </button>
  );
}
