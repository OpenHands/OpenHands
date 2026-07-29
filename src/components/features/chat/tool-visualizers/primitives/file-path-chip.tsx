import React from "react";
import FileIcon from "#/icons/file.svg?react";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { openWorkspaceFile } from "#/services/canvas-ui";

interface FilePathChipProps {
  path: string;
  /** Optional line-range suffix, e.g. "12-48". */
  range?: string;
}

/**
 * Monospace file-path pill with a file icon and an optional line-range suffix.
 * Click opens the Files drawer on this path (#16125).
 */
export function FilePathChip({ path, range }: FilePathChipProps) {
  const { conversationId } = useOptionalConversationId();
  const { data: conversation } = useActiveConversation();

  return (
    <button
      type="button"
      data-testid="file-path-chip"
      title={path}
      className="inline-flex max-w-full cursor-pointer items-center gap-1.5 self-start rounded bg-surface-raised px-2 py-0.5 text-left font-mono text-xs text-foreground hover:underline"
      onClick={(event) => {
        event.stopPropagation();
        openWorkspaceFile(
          path,
          conversationId ?? null,
          conversation?.workspace?.working_dir,
        );
      }}
    >
      <FileIcon className="h-3.5 w-3.5 flex-shrink-0 text-muted" />
      <span className="break-all">{range ? `${path}:${range}` : path}</span>
    </button>
  );
}
