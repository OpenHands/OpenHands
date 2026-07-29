import { FaFile } from "react-icons/fa";
import { RemoveButton } from "#/components/shared/buttons/remove-button";

interface FileItemProps {
  filename: string;
  onRemove?: () => void;
}

export function FileItem({ filename, onRemove }: FileItemProps) {
  return (
    <div
      data-testid="file-item"
      className="flex flex-row gap-x-1 items-center justify-start py-1"
    >
      <FaFile className="h-4 w-4" />
      <code className="text-sm flex-1 text-white truncate">{filename}</code>
      {onRemove && <RemoveButton onClick={onRemove} />}
    </div>
  );
}
