import React from "react";
import FileIcon from "#/icons/file.svg?react";

interface FilePathChipProps {
  path: string;
  /** Optional line-range suffix, e.g. "12-48". */
  range?: string;
}

/**
 * Monospace file-path pill with a file icon and an optional line-range suffix.
 */
export function FilePathChip({ path, range }: FilePathChipProps) {
  return (
    <span className="inline-flex max-w-full items-center gap-1.5 self-start rounded bg-surface-raised px-2 py-0.5 font-mono text-xs text-foreground">
      <FileIcon className="h-3.5 w-3.5 flex-shrink-0 text-muted" />
      <span className="break-all">{range ? `${path}:${range}` : path}</span>
    </span>
  );
}
