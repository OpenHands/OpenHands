import React from "react";
import { useTranslation } from "react-i18next";
import {
  ArrowUpRight,
  Copy,
  Download,
  FileCode,
  TriangleAlert,
} from "lucide-react";
import FileIcon from "#/icons/file.svg?react";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import { cn } from "#/utils/utils";

export interface InlineArtifactAction {
  key: string;
  /** Already-translated label. */
  label: string;
  icon: React.ReactNode;
  onClick: () => void;
  testId: string;
}

interface InlineArtifactCardProps {
  fileName: string;
  /** Rendered diagram/code preview body. */
  children: React.ReactNode;
  /** Extra controls shown next to View in the footer. */
  actions?: InlineArtifactAction[];
  /** Deep-link into the Files drawer; hidden when omitted (in-flight create). */
  onView?: () => void;
  /** Short, non-blocking message shown above the body (e.g. render error). */
  errorMessage?: string;
  /** Whether the preview is still being produced. */
  isLoading?: boolean;
  /** Clip the body to a compact height (chat stream) vs. fill the pane. */
  clipped?: boolean;
  testId?: string;
}

/**
 * Shared chrome for the inline artifact previews (Mermaid today, HTML/SVG via
 * the artifact-preview work in #18113): a clipped, scrollable preview body
 * with a footer that names the file and carries the actions.
 *
 * Deliberately a plain presentational component — it owns no render pipeline,
 * so the Mermaid and file-artifact previews can share the same look without
 * sharing the (very different) engines behind `children`.
 */
export function InlineArtifactCard({
  fileName,
  children,
  actions = [],
  onView,
  errorMessage,
  isLoading = false,
  clipped = true,
  testId,
}: InlineArtifactCardProps) {
  const { t } = useTranslation("openhands");

  return (
    <div
      className={cn(
        "w-full overflow-hidden rounded-xl border border-border bg-surface",
        !clipped && "flex h-full flex-col",
      )}
      data-testid={testId}
    >
      {errorMessage ? (
        <div
          className="flex items-start gap-2 border-b border-border px-4 py-2 text-xs text-danger"
          data-testid={testId ? `${testId}-error` : undefined}
          role="status"
        >
          <TriangleAlert className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
          <span className="whitespace-pre-wrap">{errorMessage}</span>
        </div>
      ) : null}
      <div
        data-testid={testId ? `${testId}-content` : undefined}
        className={cn(
          "overflow-auto px-4 py-3 text-contrast custom-scrollbar-always",
          "[--oh-scroll-fade-from:var(--oh-surface)]",
          clipped ? "max-h-40" : "flex-1",
        )}
      >
        {isLoading ? (
          <Typography.Text className="font-mono text-[11px] leading-4 tracking-[0.11px] text-muted">
            {t(I18nKey.COMMON$LOADING)}
          </Typography.Text>
        ) : (
          children
        )}
      </div>
      <div className="flex min-h-10 items-center justify-between gap-2 border-t border-border px-3 py-1.5">
        <div className="flex min-w-0 items-center gap-1.5">
          <FileIcon className="h-3.5 w-3.5 flex-shrink-0 text-muted" />
          <Typography.Text className="truncate font-mono text-[11px] leading-4 tracking-[0.11px] text-muted">
            {fileName}
          </Typography.Text>
        </div>
        <div className="flex shrink-0 items-center gap-3">
          {actions.map((action) => (
            <button
              key={action.key}
              type="button"
              onClick={action.onClick}
              data-testid={action.testId}
              className="flex cursor-pointer items-center gap-1 transition-opacity hover:opacity-80"
            >
              {action.icon}
              <Typography.Text className="text-[11px] leading-4 tracking-[0.11px] text-contrast">
                {action.label}
              </Typography.Text>
            </button>
          ))}
          {onView ? (
            <button
              type="button"
              onClick={onView}
              className="flex cursor-pointer items-center gap-1 transition-opacity hover:opacity-80"
              data-testid={testId ? `${testId}-view` : undefined}
            >
              <Typography.Text className="text-[11px] leading-4 tracking-[0.11px] text-contrast">
                {t(I18nKey.COMMON$VIEW)}
              </Typography.Text>
              <ArrowUpRight className="text-contrast" size={16} />
            </button>
          ) : null}
        </div>
      </div>
    </div>
  );
}

/** Convenience builders so callers don't re-create the action icons. */
export function copyAction(
  label: string,
  onClick: () => void,
  testId: string,
): InlineArtifactAction {
  return { key: "copy", label, onClick, testId, icon: <Copy size={14} /> };
}

export function downloadAction(
  label: string,
  onClick: () => void,
  testId: string,
): InlineArtifactAction {
  return {
    key: "download",
    label,
    onClick,
    testId,
    icon: <Download size={14} />,
  };
}

export function sourceAction(
  label: string,
  onClick: () => void,
  testId: string,
): InlineArtifactAction {
  return {
    key: "source",
    label,
    onClick,
    testId,
    icon: <FileCode size={14} />,
  };
}
