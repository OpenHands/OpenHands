import { LoaderCircle } from "lucide-react";
import FileIcon from "#/icons/file.svg?react";
import { RemoveFileButton } from "./remove-file-button";
import { cn, getFileExtension } from "#/utils/utils";

interface UploadedFileProps {
  file: File;
  onRemove: () => void;
  isLoading?: boolean;
}

export function UploadedFile({
  file,
  onRemove,
  isLoading = false,
}: UploadedFileProps) {
  const fileExtension = getFileExtension(file.name);

  return (
    <div className="group flex gap-2 rounded-lg bg-[#525252] max-w-[160px] px-3 py-1 relative">
      <div className="flex flex-col justify-center gap-0.25">
        <RemoveFileButton onClick={onRemove} />
        <div className="flex items-center gap-2 w-full">
          <span
            className={cn(
              "text-sm font-normal leading-5 flex-1 max-w-[136px] truncate",
              isLoading ? "max-w-[108px] text-[#A7A7A7]" : "text-white",
            )}
          >
            {file.name}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <FileIcon width={12} height={12} color="#A7A7A7" />
          <span className="text-[9px] font-normal leading-5 text-[#A7A7A7]">
            {fileExtension}
          </span>
        </div>
      </div>
      {isLoading && (
        <div className="flex items-center justify-center">
          <LoaderCircle className="animate-spin w-5 h-5" color="white" />
        </div>
      )}
    </div>
  );
}
