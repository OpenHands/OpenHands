interface ServerStatusContextMenuIconTextProps {
  icon: React.ReactNode;
  text: string;
  onClick: (event: React.MouseEvent<HTMLButtonElement>) => void;
  testId?: string;
}

export function ServerStatusContextMenuIconText({
  icon,
  text,
  onClick,
  testId,
}: ServerStatusContextMenuIconTextProps) {
  return (
    <button
      className="flex items-center justify-between p-2 hover:bg-[#5C5D62] rounded text-sm text-white font-normal leading-5 cursor-pointer w-full"
      onClick={onClick}
      data-testid={testId}
      type="button"
    >
      {text}
      {icon}
    </button>
  );
}
