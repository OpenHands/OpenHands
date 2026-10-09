import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import {
  copyAction,
  downloadAction,
  InlineArtifactCard,
  sourceAction,
} from "./inline-artifact-card";
import { renderMermaidToSafeSvg } from "./mermaid-renderer";

interface MermaidDiagramProps {
  /** Raw Mermaid source (the fenced block body or file contents). */
  source: string;
  /**
   * Filename shown in the card footer. Defaults to a generic label for a
   * fenced block, where there is no workspace file behind the diagram.
   */
  fileName?: string;
  /** Clip the body to a compact height (chat) vs. fill the pane (files tab). */
  clipped?: boolean;
  testId?: string;
  /**
   * Deep-links the backing workspace file into the Files drawer. Omitted for a
   * fenced block (no file behind it) and outside a conversation route; the
   * card then hides its View affordance.
   */
  onView?: () => void;
}

type RenderState =
  | { status: "loading" }
  | { status: "ready"; svg: string }
  | { status: "error" };

/** `React.useId()` yields ids like `:r0:` that are unsafe inside a DOM id. */
const toSafeDomId = (id: string) =>
  `mermaid-${id.replace(/[^a-zA-Z0-9_-]/g, "")}`;

/**
 * Renders a Mermaid diagram as an inline SVG card.
 *
 * The heavy Mermaid library is loaded through the dynamic import in
 * {@link renderMermaidToSafeSvg}, so this component's module graph stays
 * small. Rendering failure (invalid diagram) is non-fatal: the raw source is
 * shown with a short message and the rest of the message is untouched.
 */
export function MermaidDiagram({
  source,
  fileName,
  clipped = true,
  testId = "mermaid-diagram",
  onView,
}: MermaidDiagramProps) {
  const { t } = useTranslation("openhands");
  const reactId = React.useId();
  const [state, setState] = React.useState<RenderState>({ status: "loading" });
  const [showSource, setShowSource] = React.useState(false);

  React.useEffect(() => {
    let cancelled = false;
    setState({ status: "loading" });

    renderMermaidToSafeSvg(toSafeDomId(reactId), source)
      .then((svg) => {
        if (!cancelled) setState({ status: "ready", svg });
      })
      .catch(() => {
        if (!cancelled) setState({ status: "error" });
      });

    return () => {
      cancelled = true;
    };
  }, [source, reactId]);

  const svg = state.status === "ready" ? state.svg : null;
  const displaySource = showSource || state.status === "error";
  const label = fileName ?? t(I18nKey.MERMAID$DIAGRAM_LABEL);

  const handleDownload = React.useCallback(() => {
    if (!svg) return;
    const blob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${label.replace(/\.[^./\\]+$/, "") || "diagram"}.svg`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }, [svg, label]);

  const actions = [
    sourceAction(
      displaySource
        ? t(I18nKey.MERMAID$HIDE_SOURCE)
        : t(I18nKey.MERMAID$VIEW_SOURCE),
      () => setShowSource((prev) => !prev),
      `${testId}-toggle-source`,
    ),
    copyAction(
      t(I18nKey.MERMAID$COPY_SOURCE),
      () => {
        void navigator.clipboard.writeText(source);
      },
      `${testId}-copy-source`,
    ),
    downloadAction(
      t(I18nKey.MERMAID$DOWNLOAD_SVG),
      handleDownload,
      `${testId}-download-svg`,
    ),
  ];

  return (
    <InlineArtifactCard
      fileName={label}
      actions={actions}
      errorMessage={
        state.status === "error" ? t(I18nKey.MERMAID$RENDER_ERROR) : undefined
      }
      isLoading={state.status === "loading" && !displaySource}
      clipped={clipped}
      testId={testId}
      onView={onView}
    >
      {displaySource ? (
        <pre
          className="overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-4 text-muted"
          data-testid={`${testId}-source`}
        >
          {source}
        </pre>
      ) : svg ? (
        // The SVG comes from Mermaid `securityLevel: "strict"` and is run
        // through DOMPurify's SVG profile in `mermaid-renderer.ts` before it
        // reaches this sink, so it carries no scripts, event handlers,
        // `javascript:` URLs, or `foreignObject`.
        <div
          className="flex justify-center [&_svg]:h-auto [&_svg]:max-w-full"
          data-testid={`${testId}-svg`}
          dangerouslySetInnerHTML={{ __html: svg }}
        />
      ) : null}
    </InlineArtifactCard>
  );
}
