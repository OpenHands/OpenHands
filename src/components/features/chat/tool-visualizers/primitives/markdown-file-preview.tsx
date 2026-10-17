/**
 * Inline markdown artifact preview for chat tool cards.
 *
 * Shows a height-limited scrollable rich preview so long reports stay compact
 * in the stream; the footer View action deep-links into the Files drawer for
 * the full document. Markdown *create* file-editor events stay expanded and
 * ungrouped so the preview is visible without an extra click.
 */
import { useTranslation } from "react-i18next";
import { ArrowUpRight } from "lucide-react";
import FileIcon from "#/icons/file.svg?react";
import { I18nKey } from "#/i18n/declaration";
import { MarkdownRenderer } from "#/components/features/markdown/markdown-renderer";
import { planComponents } from "#/components/features/markdown/plan-components";
import { Typography } from "#/ui/typography";
import type { ActionEvent, OpenHandsEvent } from "#/types/agent-server/core";
import { isPreviewableArtifactPath } from "#/utils/is-previewable-file-path";
import {
  getFileEditorEventCommand,
  getFileEditorEventPath,
} from "./file-editor-event";

export { isMarkdownFilePath } from "#/utils/is-markdown-file-path";

interface MarkdownFilePreviewProps {
  content: string;
  path: string;
  /** When omitted (e.g. in-flight create), the View affordance is hidden. */
  onView?: () => void;
}

/**
 * True for file-editor *create* events whose path is an inline-previewable
 * artifact (Markdown, HTML, or SVG).
 *
 * Used to keep those cards expanded and outside collapsed action groups so
 * the clipped preview is visible by default. Reads/edits stay on the normal
 * groupable path.
 */
export function isPreviewableFileEditorEvent(
  event: OpenHandsEvent,
  correspondingAction?: ActionEvent,
): boolean {
  const path = getFileEditorEventPath(event, correspondingAction);
  const command = getFileEditorEventCommand(event, correspondingAction);
  return Boolean(
    path && command === "create" && isPreviewableArtifactPath(path),
  );
}

/** @deprecated Use {@link isPreviewableFileEditorEvent}. */
export const isMarkdownFileEditorEvent = isPreviewableFileEditorEvent;

/**
 * Height-clipped markdown card with an optional View bar that opens the file.
 */
export function MarkdownFilePreview({
  content,
  path,
  onView,
}: MarkdownFilePreviewProps) {
  const { t } = useTranslation("openhands");
  const fileName = path.split("/").pop() || path;

  return (
    <div
      className="w-full overflow-hidden rounded-xl border border-border bg-surface"
      data-testid="markdown-file-preview"
    >
      <div
        data-testid="markdown-file-preview-content"
        className="max-h-40 overflow-y-auto px-4 py-3 text-contrast custom-scrollbar-always [--oh-scroll-fade-from:var(--oh-surface)]"
      >
        {/* Deliberately compact: the clipped in-stream card reuses the plan
            preview's small typography; the Files drawer renders the same
            artifact full-size with the default components. */}
        <MarkdownRenderer
          content={content}
          includeStandard
          includeHeadings
          components={planComponents}
        />
      </div>
      <div className="flex h-10 items-center justify-between gap-2 border-t border-border px-3">
        <div className="flex min-w-0 items-center gap-1.5">
          <FileIcon className="h-3.5 w-3.5 flex-shrink-0 text-muted" />
          <Typography.Text className="truncate font-mono text-[11px] leading-4 tracking-[0.11px] text-muted">
            {fileName}
          </Typography.Text>
        </div>
        {onView ? (
          <button
            type="button"
            onClick={onView}
            className="flex shrink-0 cursor-pointer items-center gap-1 transition-opacity hover:opacity-80"
            data-testid="markdown-file-preview-view"
          >
            <Typography.Text className="text-[11px] leading-4 tracking-[0.11px] text-contrast">
              {t(I18nKey.COMMON$VIEW)}
            </Typography.Text>
            <ArrowUpRight className="text-contrast" size={16} />
          </button>
        ) : null}
      </div>
    </div>
  );
}
