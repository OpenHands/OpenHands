import React from "react";
import FileIcon from "#/icons/file.svg?react";

interface FilePathChipProps {
  path: string;
  /** Optional line-range suffix, e.g. "12-48". */
  range?: string;
  /** Opens the referenced file when the surrounding visualizer can navigate. */
  onClick?: () => void;
}

/**
 * Monospace file-path pill with an optional navigation affordance.
 *
 * The static variant remains available for visualizers without a workspace
 * target; passing `onClick` upgrades the same visual language to a button.
 */
export function FilePathChip({ path, range, onClick }: FilePathChipProps) {
  const content = (
    <>
      <FileIcon className="h-3.5 w-3.5 flex-shrink-0 text-muted" />
      <span className="break-all">{range ? `${path}:${range}` : path}</span>
    </>
  );
  const className =
    "inline-flex max-w-full items-center gap-1.5 self-start rounded bg-surface-raised px-2 py-0.5 font-mono text-xs text-foreground";

  if (onClick) {
    return (
      <button
        type="button"
        className={`${className} cursor-pointer hover:bg-[var(--oh-interactive-hover)]`}
        onClick={onClick}
      >
        {content}
      </button>
    );
  }

  return <span className={className}>{content}</span>;
}
