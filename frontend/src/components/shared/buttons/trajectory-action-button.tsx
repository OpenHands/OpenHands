import { Tooltip } from "@heroui/react";

interface TrajectoryActionButtonProps {
  testId?: string;
  onClick: () => void;
  icon: React.ReactNode;
  tooltip?: string;
}

export function TrajectoryActionButton({
  testId,
  onClick,
  icon,
  tooltip,
}: TrajectoryActionButtonProps) {
  const button = (
    <button
      type="button"
      data-testid={testId}
      onClick={onClick}
      className="flex items-center justify-center w-[26px] h-[26px] rounded-lg cursor-pointer bg-[#25272D] hover:bg-tertiary"
    >
      {icon}
    </button>
  );

  if (tooltip) {
    return (
      <Tooltip
        content={tooltip}
        closeDelay={100}
        className="bg-white text-black hover:bg-transparent"
      >
        {button}
      </Tooltip>
    );
  }

  return button;
}
