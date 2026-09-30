/**
 * Inline artifact preview for chat tool cards.
 *
 * HTML/SVG artifacts render inside a sandboxed iframe pointed at the workspace
 * fileserver URL, so relative assets resolve and the agent-written markup stays
 * inert (no `allow-scripts`). Raster images render as an `<img>` from the same
 * URL. PDFs render in an unsandboxed iframe: a sandboxed frame is not allowed
 * to instantiate a plugin, so Chromium's built-in viewer would never appear.
 * The frame/image is mounted only while the card is near the viewport and
 * unmounted again once it leaves, so a long conversation does not keep every
 * visited frame alive.
 */
import React from "react";
import { useTranslation } from "react-i18next";
import {
  ArrowUpRight,
  Check,
  Copy,
  Download,
  Maximize2,
  Minimize2,
} from "lucide-react";
import FileIcon from "#/icons/file.svg?react";
import { I18nKey } from "#/i18n/declaration";
import { useWorkspaceFileContent } from "#/hooks/query/use-workspace-file-content";
import {
  useWorkspaceMutationCounter,
  withWorkspaceCacheBuster,
} from "#/stores/use-workspace-mutation-counter";
import { Typography } from "#/ui/typography";
import { getArtifactPreviewKind } from "#/utils/is-previewable-file-path";
import { cn } from "#/utils/utils";

interface ArtifactPreviewProps {
  path: string;
  /**
   * Workspace-relative path used to fetch the file. Defaults to `path`, but
   * callers pass the converted form when the event path is absolute (rooted at
   * the conversation's working dir). `path` itself stays the display name.
   */
  sourcePath?: string;
  /** Source text of the artifact, used for Copy. */
  content: string;
  /** When omitted (e.g. in-flight create), the View affordance is hidden. */
  onView?: () => void;
}

/**
 * Height-clipped live preview of an HTML/SVG, image, or PDF artifact, with
 * Expand / View / Copy / Download actions.
 */
export function ArtifactPreview({
  path,
  sourcePath,
  content,
  onView,
}: ArtifactPreviewProps) {
  const { t } = useTranslation("openhands");
  const fileName = path.split("/").pop() || path;
  const kind = getArtifactPreviewKind(path);
  const fetchPath = sourcePath ?? path;

  const [expanded, setExpanded] = React.useState(false);
  const [inView, setInView] = React.useState(false);
  const [copied, setCopied] = React.useState(false);
  const placeholderRef = React.useRef<HTMLDivElement>(null);

  const query = useWorkspaceFileContent(fetchPath);
  // Refetch the frame after every agent-side edit so a rewrite of this file
  // (or a sibling asset it references) is reflected without a manual reload.
  const mutationCounter = useWorkspaceMutationCounter((state) => state.count);
  const staticUrl = query.data?.staticUrl
    ? withWorkspaceCacheBuster(query.data.staticUrl, mutationCounter)
    : null;

  // Mount the frame only while the card is near the viewport, and unmount it
  // again once it leaves. Keeping the observer alive (rather than disconnecting
  // on first intersection) means a long conversation does not accumulate live
  // frames for every card the user has scrolled past.
  React.useEffect(() => {
    const node = placeholderRef.current;
    if (!node) return undefined;
    // Older engines (and jsdom) have no IntersectionObserver; fall back to
    // mounting immediately rather than never rendering the frame.
    if (typeof IntersectionObserver === "undefined") {
      setInView(true);
      return undefined;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        // 600px of padding above and below the viewport so a frame is ready
        // just before it scrolls in, without keeping it alive far off-screen.
        setInView(entries.some((entry) => entry.isIntersecting));
      },
      { rootMargin: "600px 0px 600px 0px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const copySource = React.useCallback(async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard can be unavailable (insecure context / denied); leave the
      // button in its idle state rather than surfacing an error toast.
    }
  }, [content]);

  const download = React.useCallback(async () => {
    if (!staticUrl) return;
    // An anchor's `download` attribute is ignored for cross-origin HTTP URLs
    // (Canvas and the workspace fileserver can be different origins), which
    // would navigate to the artifact instead of saving it. Fetch the bytes and
    // hand the browser a same-origin blob URL instead.
    if (!staticUrl.startsWith("data:")) {
      try {
        const response = await fetch(staticUrl, { credentials: "include" });
        if (response.ok) {
          const blob = await response.blob();
          const objectUrl = URL.createObjectURL(blob);
          const anchor = document.createElement("a");
          anchor.href = objectUrl;
          anchor.download = fileName;
          anchor.rel = "noopener";
          anchor.click();
          URL.revokeObjectURL(objectUrl);
          return;
        }
      } catch {
        // Network/CORS failure: fall back to the plain anchor below.
      }
    }
    const anchor = document.createElement("a");
    anchor.href = staticUrl;
    anchor.download = fileName;
    anchor.rel = "noopener";
    anchor.click();
  }, [staticUrl, fileName]);

  return (
    <div
      className="w-full overflow-hidden rounded-xl border border-border bg-surface"
      data-testid="artifact-preview"
    >
      <div
        ref={placeholderRef}
        data-testid="artifact-preview-frame-container"
        className={cn(
          "w-full overflow-hidden bg-white",
          expanded ? "h-[32rem]" : "h-40",
        )}
      >
        {staticUrl && inView ? (
          kind === "image" ? (
            <img
              src={staticUrl}
              alt={fileName}
              data-testid="artifact-preview-image"
              className="h-full w-full object-contain"
            />
          ) : (
            <iframe
              title={path}
              src={staticUrl}
              // HTML/SVG: `allow-same-origin` keeps the frame on the workspace
              // fileserver origin so relative `<link>` / `<img>` resolve; the
              // absence of `allow-scripts` keeps agent-written `<script>` and
              // inline handlers inert. Mirrors FileContentViewer's posture.
              //
              // PDF: Chromium refuses to instantiate the PDF plugin inside a
              // sandboxed frame, so the PDF viewer needs no sandbox. The file
              // is still the agent's own artifact on the workspace origin, and
              // no script runs from a PDF.
              sandbox={kind === "pdf" ? undefined : "allow-same-origin"}
              data-testid={
                kind === "pdf"
                  ? "artifact-preview-pdf-frame"
                  : "artifact-preview-frame"
              }
              className="h-full w-full"
            />
          )
        ) : (
          <div
            data-testid="artifact-preview-pending"
            className="flex h-full w-full items-center justify-center text-xs text-muted"
          >
            {query.isError
              ? t(I18nKey.FILES$LOAD_ERROR)
              : t(I18nKey.FILES$LOADING_FILES)}
          </div>
        )}
      </div>
      <div className="flex h-10 items-center justify-between gap-2 border-t border-border px-3">
        <div className="flex min-w-0 items-center gap-1.5">
          <FileIcon className="h-3.5 w-3.5 flex-shrink-0 text-muted" />
          <Typography.Text className="truncate font-mono text-[11px] leading-4 tracking-[0.11px] text-muted">
            {fileName}
          </Typography.Text>
        </div>
        <div className="flex shrink-0 items-center gap-3">
          <button
            type="button"
            onClick={() => setExpanded((value) => !value)}
            aria-expanded={expanded}
            className="flex cursor-pointer items-center gap-1 text-contrast transition-opacity hover:opacity-80"
            data-testid="artifact-preview-expand"
          >
            {expanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
            <Typography.Text className="text-[11px] leading-4 tracking-[0.11px] text-contrast">
              {expanded ? t(I18nKey.BUTTON$COLLAPSE) : t(I18nKey.BUTTON$EXPAND)}
            </Typography.Text>
          </button>
          <button
            type="button"
            onClick={copySource}
            className="flex cursor-pointer items-center gap-1 text-contrast transition-opacity hover:opacity-80"
            data-testid="artifact-preview-copy"
          >
            {copied ? <Check size={14} /> : <Copy size={14} />}
            <Typography.Text className="text-[11px] leading-4 tracking-[0.11px] text-contrast">
              {t(I18nKey.FEEDBACK$COPY_LABEL)}
            </Typography.Text>
          </button>
          <button
            type="button"
            onClick={download}
            disabled={!staticUrl}
            aria-label={t(I18nKey.BUTTON$DOWNLOAD)}
            title={t(I18nKey.BUTTON$DOWNLOAD)}
            className="flex cursor-pointer items-center gap-1 text-contrast transition-opacity hover:opacity-80 disabled:cursor-not-allowed disabled:opacity-50"
            data-testid="artifact-preview-download"
          >
            <Download size={14} />
          </button>
          {onView ? (
            <button
              type="button"
              onClick={onView}
              className="flex cursor-pointer items-center gap-1 transition-opacity hover:opacity-80"
              data-testid="artifact-preview-view"
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
