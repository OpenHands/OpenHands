import React from "react";
import { useColorTheme } from "#/hooks/use-color-theme";
import { COLOR_THEMES } from "#/themes/color-themes";

interface MermaidDiagramProps {
  /** The Mermaid source from a ```mermaid fenced code block. */
  source: string;
  /** Shown instead of the diagram when the source is invalid. */
  fallback: React.ReactNode;
}

type RenderState =
  | { status: "pending" }
  | { status: "rendered"; svg: string }
  | { status: "invalid" };

let nextDiagramId = 0;

// Rendered SVGs by theme and source. Rendering is deterministic for a given
// pair, so a remount (navigating back, a re-rendered chat message) shows the
// diagram at once instead of loading Mermaid and laying it out again.
const MAX_CACHED_DIAGRAMS = 50;
const svgCache = new Map<string, string>();

function cacheKey(isDark: boolean, source: string) {
  return `${isDark ? "dark" : "light"}\n${source}`;
}

function remember(key: string, svg: string) {
  svgCache.delete(key);
  svgCache.set(key, svg);
  if (svgCache.size > MAX_CACHED_DIAGRAMS) {
    svgCache.delete(svgCache.keys().next().value!);
  }
}

function initialState(key: string): RenderState {
  const svg = svgCache.get(key);
  return svg === undefined
    ? { status: "pending" }
    : { status: "rendered", svg };
}

/**
 * Renders a ```mermaid fenced code block as a diagram.
 *
 * Mermaid is large, so it is imported on the first diagram rather than with
 * the app. The source is untrusted (chat replies, automation descriptions),
 * so it renders with `securityLevel: "strict"`: Mermaid sanitizes labels and
 * disables click handlers. While the first render is in flight a placeholder
 * holds the space; invalid source, including a streamed block that is not
 * finished yet, shows the fallback (the plain code block) instead.
 */
export function MermaidDiagram({ source, fallback }: MermaidDiagramProps) {
  const colorTheme = useColorTheme();
  const isDark = COLOR_THEMES[colorTheme]?.appearance === "dark";
  const key = cacheKey(isDark, source);
  const [state, setState] = React.useState<RenderState>(() =>
    initialState(key),
  );

  React.useEffect(() => {
    const cached = svgCache.get(key);
    if (cached !== undefined) {
      setState({ status: "rendered", svg: cached });
      return undefined;
    }

    let cancelled = false;

    (async () => {
      const { default: mermaid } = await import("mermaid");
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: "strict",
        theme: isDark ? "dark" : "default",
      });
      // `parse` first: `render` on invalid source leaves an error graphic
      // in the document body.
      const isValid = await mermaid.parse(source, { suppressErrors: true });
      if (!isValid) {
        if (!cancelled) setState({ status: "invalid" });
        return;
      }
      nextDiagramId += 1;
      const { svg } = await mermaid.render(`mermaid-${nextDiagramId}`, source);
      remember(key, svg);
      if (!cancelled) setState({ status: "rendered", svg });
    })().catch(() => {
      if (!cancelled) setState({ status: "invalid" });
    });

    return () => {
      cancelled = true;
    };
  }, [key, source, isDark]);

  if (state.status === "invalid") {
    return fallback;
  }

  if (state.status === "pending") {
    return (
      <div
        data-testid="mermaid-diagram-pending"
        aria-busy
        className="h-40 animate-pulse rounded-lg border border-surface-raised bg-surface-raised/40"
      />
    );
  }

  return (
    <div
      data-testid="mermaid-diagram"
      // Cap the height so a tall diagram scales down (the SVG keeps its
      // aspect ratio) instead of filling the page.
      className="flex justify-center overflow-x-auto rounded-lg border border-surface-raised p-4 [&_svg]:max-h-96"
      // The SVG comes from Mermaid in strict mode, which sanitizes it.
      dangerouslySetInnerHTML={{ __html: state.svg }}
    />
  );
}
