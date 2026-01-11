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
      className="button-base p-1 hover:bg-surface-hover rotate-180 cursor-pointer"
    >
      <ArrowSendIcon width={15} height={15} />
    </button>
  );
}
