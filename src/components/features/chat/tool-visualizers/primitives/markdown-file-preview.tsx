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
import { isMarkdownFilePath } from "#/utils/is-markdown-file-path";
import type {
  ActionEvent,
  ObservationEvent,
  OpenHandsEvent,
} from "#/types/agent-server/core";
import {
  isActionEvent,
  isObservationEvent,
} from "#/types/agent-server/type-guards";
import type {
  FileEditorAction,
  StrReplaceEditorAction,
} from "#/types/agent-server/core/base/action";
import type {
  FileEditorObservation,
  StrReplaceEditorObservation,
} from "#/types/agent-server/core/base/observation";

export { isMarkdownFilePath } from "#/utils/is-markdown-file-path";

const FILE_EDITOR_ACTION_KINDS = new Set([
  "FileEditorAction",
  "StrReplaceEditorAction",
]);
const FILE_EDITOR_OBSERVATION_KINDS = new Set([
  "FileEditorObservation",
  "StrReplaceEditorObservation",
]);

interface MarkdownFilePreviewProps {
  content: string;
  path: string;
  /** When omitted (e.g. in-flight create), the View affordance is hidden. */
  onView?: () => void;
}

/**
 * Resolves the file path for a file-editor action/observation, including the
 * observation's originating action when the observation omits `path`.
 */
function getFileEditorEventPath(
  event: OpenHandsEvent,
  correspondingAction?: ActionEvent,
): string | null {
  if (isActionEvent(event) && FILE_EDITOR_ACTION_KINDS.has(event.action.kind)) {
    return (
      (event as ActionEvent<FileEditorAction | StrReplaceEditorAction>).action
        .path || null
    );
  }

  if (
    isObservationEvent(event) &&
    FILE_EDITOR_OBSERVATION_KINDS.has(event.observation.kind)
  ) {
    const path = (
      event as ObservationEvent<
        FileEditorObservation | StrReplaceEditorObservation
      >
    ).observation.path;
    if (path) return path;
    if (
      correspondingAction &&
      FILE_EDITOR_ACTION_KINDS.has(correspondingAction.action.kind)
    ) {
      return (
        (
          correspondingAction as ActionEvent<
            FileEditorAction | StrReplaceEditorAction
          >
        ).action.path || null
      );
    }
  }

  return null;
}

/**
 * Resolves the file-editor command (`create` / `view` / …), including the
 * observation's originating action when the observation omits `command`.
 */
function getFileEditorEventCommand(
  event: OpenHandsEvent,
  correspondingAction?: ActionEvent,
): string | null {
  if (isActionEvent(event) && FILE_EDITOR_ACTION_KINDS.has(event.action.kind)) {
    return (
      (event as ActionEvent<FileEditorAction | StrReplaceEditorAction>).action
        .command || null
    );
  }

  if (
    isObservationEvent(event) &&
    FILE_EDITOR_OBSERVATION_KINDS.has(event.observation.kind)
  ) {
    const command = (
      event as ObservationEvent<
        FileEditorObservation | StrReplaceEditorObservation
      >
    ).observation.command;
    if (command) return command;
    if (
      correspondingAction &&
      FILE_EDITOR_ACTION_KINDS.has(correspondingAction.action.kind)
    ) {
      return (
        (
          correspondingAction as ActionEvent<
            FileEditorAction | StrReplaceEditorAction
          >
        ).action.command || null
      );
    }
  }

  return null;
}

/**
 * True for file-editor *create* events whose path is a markdown artifact.
 *
 * Used to keep those cards expanded and outside collapsed action groups so
 * the clipped preview is visible by default. Reads/edits of `.md` files stay
 * on the normal groupable path.
 */
export function isMarkdownFileEditorEvent(
  event: OpenHandsEvent,
  correspondingAction?: ActionEvent,
): boolean {
  const path = getFileEditorEventPath(event, correspondingAction);
  const command = getFileEditorEventCommand(event, correspondingAction);
  return Boolean(path && command === "create" && isMarkdownFilePath(path));
}

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
      className="w-full overflow-hidden rounded-[12px] border border-[var(--oh-border)] bg-[var(--oh-surface)]"
      data-testid="markdown-file-preview"
    >
      <div
        data-testid="markdown-file-preview-content"
        className="max-h-40 overflow-y-auto px-4 py-3 text-white custom-scrollbar-always [--oh-scroll-fade-from:var(--oh-surface)]"
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
      <div className="flex h-10 items-center justify-between gap-2 border-t border-[var(--oh-border)] px-3">
        <div className="flex min-w-0 items-center gap-1.5">
          <FileIcon className="h-3.5 w-3.5 flex-shrink-0 text-[var(--oh-muted)]" />
          <Typography.Text className="truncate font-mono text-[11px] leading-4 tracking-[0.11px] text-[var(--oh-muted)]">
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
            <Typography.Text className="text-[11px] leading-4 tracking-[0.11px] text-white">
              {t(I18nKey.COMMON$VIEW)}
            </Typography.Text>
            <ArrowUpRight className="text-white" size={16} />
          </button>
        ) : null}
      </div>
    </div>
  );
}
