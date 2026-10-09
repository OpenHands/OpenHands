/**
 * Inline preview for Office documents (`.docx` / `.xlsx` / `.pptx`).
 *
 * The bytes are unpacked client-side by `readOoxmlPreview` — no renderer
 * dependency and no server round-trip beyond the workspace file the card
 * already points at. Parsing waits until the card scrolls into view, so a long
 * conversation with many documents does not unpack every one of them.
 *
 * The render is an outline (headings, sheets, slides), not a pixel-faithful
 * reproduction: the point is to let a reviewer see what the document says
 * without leaving the conversation.
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
import { getFileExtension } from "#/utils/is-previewable-file-path";
import {
  readOoxmlPreview,
  type OoxmlKind,
  type OoxmlPreview,
} from "#/utils/ooxml-preview";
import { cn } from "#/utils/utils";

interface OfficeArtifactPreviewProps {
  path: string;
  /**
   * Workspace-relative path used to fetch the file. Defaults to `path`, but
   * callers pass the converted form when the event path is absolute (rooted at
   * the conversation's working dir). `path` itself stays the display name.
   */
  sourcePath?: string;
  /** Source text of the artifact, used for Copy. */
  content?: string;
  /** When omitted (e.g. in-flight create), the View affordance is hidden. */
  onView?: () => void;
}

const KIND_BY_EXTENSION: Record<string, OoxmlKind> = {
  docx: "docx",
  xlsx: "xlsx",
  pptx: "pptx",
};

/** Word body: headings stand out, list items keep their marker. */
function WordOutline({ blocks }: { blocks: OoxmlPreview["blocks"] }) {
  return (
    <div className="flex flex-col gap-1.5">
      {blocks.map((block, index) => {
        if (block.type === "heading") {
          return (
            <Typography.Text
              key={index}
              className="mt-1 text-sm font-semibold text-contrast"
            >
              {block.text}
            </Typography.Text>
          );
        }
        if (block.type === "list-item") {
          return (
            <Typography.Text
              key={index}
              className="flex gap-1.5 text-xs leading-5 text-contrast"
            >
              <span aria-hidden className="text-muted">
                {block.ordered ? "•" : "–"}
              </span>
              <span>{block.text}</span>
            </Typography.Text>
          );
        }
        return (
          <Typography.Text
            key={index}
            className="text-xs leading-5 text-contrast"
          >
            {block.text}
          </Typography.Text>
        );
      })}
    </div>
  );
}

/** Excel: a sheet tab strip over the selected sheet's grid. */
function SheetGrid({ sheets }: { sheets: OoxmlPreview["sheets"] }) {
  const [selected, setSelected] = React.useState(0);
  const sheet = sheets[selected];
  if (!sheet) return null;
  const [header, ...body] = sheet.rows;

  return (
    <div className="flex flex-col gap-2">
      {sheets.length > 1 ? (
        <div className="flex flex-wrap gap-1" data-testid="office-sheet-tabs">
          {sheets.map((candidate, index) => (
            <button
              key={candidate.name}
              type="button"
              onClick={() => setSelected(index)}
              data-testid={`office-sheet-tab-${index}`}
              data-state={index === selected ? "active" : "inactive"}
              className={cn(
                "cursor-pointer rounded px-1.5 py-0.5 text-xs",
                index === selected
                  ? "bg-interactive-hover text-contrast"
                  : "text-muted hover:opacity-80",
              )}
            >
              {candidate.name}
            </button>
          ))}
        </div>
      ) : null}
      <div className="overflow-x-auto">
        <table
          data-testid="office-sheet-table"
          className="w-full border-collapse text-xs"
        >
          {header ? (
            <thead>
              <tr>
                {header.map((cell, index) => (
                  <th
                    key={index}
                    className="border border-border px-1.5 py-1 text-left font-semibold text-contrast"
                  >
                    {cell}
                  </th>
                ))}
              </tr>
            </thead>
          ) : null}
          <tbody>
            {body.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {row.map((cell, cellIndex) => (
                  <td
                    key={cellIndex}
                    className="border border-border px-1.5 py-1 text-contrast"
                  >
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** PowerPoint: one block per slide, numbered by position. */
function SlideOutline({ slides }: { slides: OoxmlPreview["slides"] }) {
  return (
    <div className="flex flex-col gap-3">
      {slides.map((slide) => (
        <div
          key={slide.index}
          data-testid={`office-slide-${slide.index}`}
          className="flex flex-col gap-1"
        >
          <Typography.Text className="flex gap-1.5 text-xs text-muted">
            <span className="font-mono">{slide.index}</span>
            {slide.title ? (
              <span className="font-semibold text-contrast">{slide.title}</span>
            ) : null}
          </Typography.Text>
          {slide.lines.map((line, index) => (
            <Typography.Text
              key={index}
              className="pl-4 text-xs leading-5 text-contrast"
            >
              {line}
            </Typography.Text>
          ))}
        </div>
      ))}
    </div>
  );
}

/**
 * Height-clipped outline of an Office document, with Expand / View / Copy /
 * Download actions — the same control set the other artifact cards offer.
 */
export function OfficeArtifactPreview({
  path,
  sourcePath,
  content,
  onView,
}: OfficeArtifactPreviewProps) {
  const { t } = useTranslation("openhands");
  const fileName = path.split("/").pop() || path;
  const kind = KIND_BY_EXTENSION[getFileExtension(path)];
  const fetchPath = sourcePath ?? path;

  const [expanded, setExpanded] = React.useState(false);
  const [inView, setInView] = React.useState(false);
  // Both results are tagged with the `staticUrl` they were produced from, so a
  // changed URL invalidates them without a separate reset effect (which would
  // flash the loading state on every mutation-counter tick).
  const [preview, setPreview] = React.useState<{
    url: string;
    value: OoxmlPreview;
  } | null>(null);
  const [failure, setFailure] = React.useState<{
    url: string;
    reason: "error" | "too-large" | "unavailable";
  } | null>(null);
  const [copied, setCopied] = React.useState(false);
  const containerRef = React.useRef<HTMLDivElement>(null);

  const query = useWorkspaceFileContent(fetchPath);
  // Refetch after an agent-side rewrite so the outline reflects the new bytes.
  const mutationCounter = useWorkspaceMutationCounter((state) => state.count);
  const staticUrl = query.data?.staticUrl
    ? withWorkspaceCacheBuster(query.data.staticUrl, mutationCounter)
    : null;

  React.useEffect(() => {
    const node = containerRef.current;
    if (!node || inView) return undefined;
    // Older engines (and jsdom) have no IntersectionObserver; fall back to
    // parsing immediately rather than never rendering the outline.
    if (typeof IntersectionObserver === "undefined") {
      setInView(true);
      return undefined;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) setInView(true);
      },
      { rootMargin: "200px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [inView]);

  const copySource = React.useCallback(async () => {
    if (content == null) return;
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
    // `download` is ignored for cross-origin HTTP URLs, so fetch the bytes and
    // save a same-origin blob URL instead (see ArtifactPreview for the detail).
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

  React.useEffect(() => {
    // Re-parse whenever the cache-busted URL changes. The result is tagged with
    // the URL it came from, so an agent-side rewrite (which bumps the mutation
    // counter) clears a previous outline *or* a previous error instead of being
    // skipped — otherwise a document first seen as "Draft" keeps showing
    // "Draft" after the agent writes "Final", and a 404 stays a 404 after the
    // file is created.
    if (!inView || !staticUrl || !kind || !query.data) return undefined;
    if (preview?.url === staticUrl || failure?.url === staticUrl)
      return undefined;
    let cancelled = false;
    (async () => {
      try {
        // Cloud serves file content as a UTF-8-decoded *string*, so a binary
        // container's ZIP bytes are already destroyed and re-encoding them
        // cannot restore them. Withhold the outline rather than showing a
        // parse error for a document that is fine — the card still offers
        // Download (and View), which are the useful actions there.
        if (query.data.bytesLossy) {
          if (!cancelled) setFailure({ url: staticUrl, reason: "unavailable" });
          return;
        }
        // Oversized: the hook refused the body on its declared length, so there
        // are no bytes to parse and the reason is size, not a parse failure.
        if (query.data.bytesTooLarge) {
          if (!cancelled) setFailure({ url: staticUrl, reason: "too-large" });
          return;
        }
        // Parse the bytes the file-content hook fetched — the hook owns the
        // (typed, bounded, cache-busted) transport, so the card never downloads
        // the document a second time over a raw fetch. `bytes` is always
        // populated for a local binary file; a null `bytes` with no size flag
        // means the transport could not supply them (e.g. a Cloud data: URI),
        // which is reported as a plain load failure.
        if (query.data.bytes == null) {
          if (!cancelled) setFailure({ url: staticUrl, reason: "error" });
          return;
        }
        // The bytes must belong to the version this URL names. After an edit the
        // query refetches under the new counter, but `query.data` still holds
        // the previous version's bytes until it resolves; parsing those and
        // tagging them with the new URL would show a stale outline. Wait for the
        // refetched bytes instead.
        if (query.data.bytesVersion !== mutationCounter) return;
        const parsed = await readOoxmlPreview(kind, query.data.bytes);
        if (!cancelled) setPreview({ url: staticUrl, value: parsed });
      } catch (error) {
        if (cancelled) return;
        // A size rejection is actionable ("too large"), not a generic load
        // failure, so surface the distinction rather than a misleading error.
        setFailure({
          url: staticUrl,
          reason:
            error instanceof Error && /exceeds|too large/i.test(error.message)
              ? "too-large"
              : "error",
        });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [inView, staticUrl, kind, preview, failure, query.data]);

  return (
    <div
      className="w-full overflow-hidden rounded-xl border border-border bg-surface"
      data-testid="office-artifact-preview"
    >
      <div
        ref={containerRef}
        data-testid="office-artifact-preview-content"
        className={cn(
          "overflow-auto px-4 py-3 custom-scrollbar-always [--oh-scroll-fade-from:var(--oh-surface)]",
          expanded ? "max-h-[32rem]" : "max-h-48",
        )}
      >
        {failure?.url === staticUrl ||
        (query.isError && preview?.url !== staticUrl) ? (
          <Typography.Text
            className="text-xs text-muted"
            testId="office-artifact-preview-error"
          >
            {failure?.reason === "too-large"
              ? t(I18nKey.FILES$FILE_TOO_LARGE)
              : failure?.reason === "unavailable"
                ? t(I18nKey.FILES$BINARY_FALLBACK)
                : t(I18nKey.FILES$LOAD_ERROR)}
          </Typography.Text>
        ) : preview?.url !== staticUrl ? (
          <Typography.Text
            className="text-xs text-muted"
            testId="office-artifact-preview-pending"
          >
            {t(I18nKey.FILES$LOADING_FILES)}
          </Typography.Text>
        ) : (
          <>
            {preview.value.kind === "docx" ? (
              <WordOutline blocks={preview.value.blocks} />
            ) : null}
            {preview.value.kind === "xlsx" ? (
              <SheetGrid sheets={preview.value.sheets} />
            ) : null}
            {preview.value.kind === "pptx" ? (
              <SlideOutline slides={preview.value.slides} />
            ) : null}
            {preview.value.truncated ? (
              <Typography.Text className="mt-2 block text-xs text-muted">
                …
              </Typography.Text>
            ) : null}
          </>
        )}
      </div>
      <div className="flex h-10 items-center justify-between gap-2 border-t border-border px-3">
        <div className="flex min-w-0 items-center gap-1.5">
          <FileIcon className="h-3.5 w-3.5 flex-shrink-0 text-muted" />
          <Typography.Text className="truncate font-mono text-xs leading-4 tracking-[0.11px] text-muted">
            {fileName}
          </Typography.Text>
        </div>
        <div className="flex shrink-0 items-center gap-3">
          <button
            type="button"
            onClick={() => setExpanded((value) => !value)}
            aria-expanded={expanded}
            className="flex cursor-pointer items-center gap-1 text-contrast transition-opacity hover:opacity-80"
            data-testid="office-artifact-preview-expand"
          >
            {expanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
            <Typography.Text className="text-xs leading-4 tracking-[0.11px] text-contrast">
              {expanded ? t(I18nKey.BUTTON$COLLAPSE) : t(I18nKey.BUTTON$EXPAND)}
            </Typography.Text>
          </button>
          <button
            type="button"
            onClick={copySource}
            disabled={content == null}
            className="flex cursor-pointer items-center gap-1 text-contrast transition-opacity hover:opacity-80 disabled:cursor-not-allowed disabled:opacity-50"
            data-testid="office-artifact-preview-copy"
          >
            {copied ? <Check size={14} /> : <Copy size={14} />}
            <Typography.Text className="text-xs leading-4 tracking-[0.11px] text-contrast">
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
            data-testid="office-artifact-preview-download"
          >
            <Download size={14} />
          </button>
          {onView ? (
            <button
              type="button"
              onClick={onView}
              className="flex cursor-pointer items-center gap-1 transition-opacity hover:opacity-80"
              data-testid="office-artifact-preview-view"
            >
              <Typography.Text className="text-xs leading-4 tracking-[0.11px] text-contrast">
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
