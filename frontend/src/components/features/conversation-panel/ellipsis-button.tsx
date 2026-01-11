import ThreeDotsVerticalIcon from "#/icons/three-dots-vertical.svg?react";

interface EllipsisButtonProps {
  onClick: (event: React.MouseEvent<HTMLButtonElement>) => void;
  fill?: string;
}

export function EllipsisButton({
  onClick,
  fill = "var(--color-muted)",
}: EllipsisButtonProps) {
  return (
    <button
      data-testid="ellipsis-button"
      type="button"
      onClick={onClick}
      className="cursor-pointer"
    >
      <ThreeDotsVerticalIcon width={24} height={24} color={fill} />
    </button>
  );
}
