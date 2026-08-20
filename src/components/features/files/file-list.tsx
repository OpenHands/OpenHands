import React from "react";
import { cn } from "#/utils/utils";
import { FileItem } from "./file-item";

interface FileListProps {
  files: string[];
  onRemove?: (index: number) => void;
}

export function FileList({ files, onRemove }: FileListProps) {
  return (
    <div
      data-testid="file-list"
      className={cn("flex flex-col gap-y-1.5 justify-start")}
    >
      {files.map((f, index) => (
        <FileItem
          key={index}
          filename={f}
          onRemove={onRemove ? () => onRemove?.(index) : undefined}
        />
      ))}
    </div>
  );
}
