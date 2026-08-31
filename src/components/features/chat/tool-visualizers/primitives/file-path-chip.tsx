import FileIcon from "#/icons/file.svg?react";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { openWorkspaceFile } from "#/services/canvas-ui";

interface FilePathChipProps {
  path: string;
  /** Optional line-range suffix, e.g. "12-48". */
  range?: string;
  /**
   * Opens the referenced file when the surrounding visualizer can navigate.
   * Defaults to `openWorkspaceFile` so chips without a parent handler still
   * deep-link into the Files drawer.
   */
  onClick?: () => void;
}

/**
 * Monospace file-path pill with an optional navigation affordance.
 *
 * Passing `onClick` upgrades the same visual language to a custom handler
 * (e.g. markdown artifact View). Without it, click opens the Files drawer.
 */
export function FilePathChip({ path, range, onClick }: FilePathChipProps) {
  const { conversationId } = useOptionalConversationId();
  const content = (
    <>
      <FileIcon className="h-3.5 w-3.5 flex-shrink-0 text-muted" />
      <span className="break-all">{range ? `${path}:${range}` : path}</span>
    </>
  );
  const className =
    "inline-flex max-w-full cursor-pointer items-center gap-1.5 self-start rounded bg-surface-raised px-2 py-0.5 text-left font-mono text-xs text-foreground hover:bg-[var(--oh-interactive-hover)]";

  return (
    <button
      type="button"
      data-testid="file-path-chip"
      title={path}
      className={className}
      onClick={(event) => {
        event.stopPropagation();
        if (onClick) {
          onClick();
          return;
        }
        openWorkspaceFile(path, conversationId);
      }}
    >
      {content}
    </button>
  );
}
