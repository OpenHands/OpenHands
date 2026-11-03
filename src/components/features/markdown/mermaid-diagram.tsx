import React from "react";
import type { MermaidConfig } from "mermaid";
import { useColorTheme } from "#/hooks/use-color-theme";
import { COLOR_THEMES } from "#/themes/color-themes";
import { cn } from "#/utils/utils";

interface MermaidDiagramProps {
  /** The Mermaid source from a ```mermaid fenced code block. */
  source: string;
  /** Shown instead of the diagram when the source is invalid. */
  fallback: React.ReactNode;
  className?: string;
}

type RenderState =
  | { status: "pending" }
  | { status: "rendered"; svg: string }
  | { status: "invalid" };

let nextDiagramId = 0;

// Rendered SVGs by color theme and source. Rendering is deterministic for a
// given pair, so a remount (navigating back, a re-rendered chat message)
// shows the diagram at once instead of loading Mermaid and laying it out again.
const MAX_CACHED_DIAGRAMS = 50;
const svgCache = new Map<string, string>();

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
 * Mermaid theme variables taken from the classes the rest of the UI uses, so
 * a node looks like a raised card in whichever color theme is active. Mermaid
 * needs concrete colors, so the classes are applied to a probe element inside
 * `host` (the app's CSS is scoped to its shell) and read back computed.
 */
function canvasThemeVariables(
  host: HTMLElement,
  isDark: boolean,
): NonNullable<MermaidConfig["themeVariables"]> {
  const probe = document.createElement("span");
  probe.hidden = true;
  host.appendChild(probe);
  const read = (className: string, property: "color" | "backgroundColor") => {
    probe.className = className;
    return getComputedStyle(probe)[property];
  };
  const card = read("bg-surface-raised", "backgroundColor");
  const page = read("bg-surface", "backgroundColor");
  const border = read("text-border", "color");
  const text = read("text-content", "color");
  const muted = read("text-muted", "color");
  probe.remove();

  // The UI font, read from the app shell: a markdown code block sits inside a
  // `<pre>`, which would otherwise lend the diagram its monospace font.
  const shell = host.closest("[data-agent-server-ui]") ?? document.body;

  return {
    darkMode: isDark,
    fontFamily: getComputedStyle(shell).fontFamily,
    fontSize: "13px",
    background: page,
    primaryColor: card,
    primaryBorderColor: border,
    primaryTextColor: text,
    secondaryColor: card,
    secondaryBorderColor: border,
    secondaryTextColor: text,
    tertiaryColor: card,
    tertiaryBorderColor: border,
    tertiaryTextColor: text,
    mainBkg: card,
    nodeBorder: border,
    nodeTextColor: text,
    textColor: text,
    lineColor: muted,
    edgeLabelBackground: page,
    clusterBkg: page,
    clusterBorder: border,
    titleColor: text,
  };
}

// Rounded nodes and lighter edge labels, matching the card corners elsewhere.
const THEME_CSS = `
  .node rect, .node polygon, .node circle, .node path { stroke-width: 1px; }
  .node rect { rx: 8px; ry: 8px; }
  .edgeLabel, .edgeLabel p { font-size: 12px; }
`;

/**
 * Renders a ```mermaid fenced code block as a diagram.
 *
 * Mermaid is large, so it is imported on the first diagram rather than with
 * the app. The source is untrusted (chat replies, automation descriptions,
 * catalog entries), so it renders with `securityLevel: "strict"`: Mermaid
 * sanitizes labels and disables click handlers. Colors and font come from the
 * active Canvas theme. While the first render is in flight a placeholder
 * holds the space; invalid source, including a streamed block that is not
 * finished yet, shows the fallback instead.
 */
export function MermaidDiagram({
  source,
  fallback,
  className,
}: MermaidDiagramProps) {
  const colorTheme = useColorTheme();
  const isDark = COLOR_THEMES[colorTheme]?.appearance === "dark";
  const key = `${colorTheme}\n${source}`;
  const hostRef = React.useRef<HTMLDivElement>(null);
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
      const host = hostRef.current;
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: "strict",
        theme: "base",
        themeVariables: host
          ? canvasThemeVariables(host, isDark)
          : { darkMode: isDark },
        themeCSS: THEME_CSS,
        // Tighter than Mermaid's defaults so a small diagram fits a panel or
        // dialog at full text size; the wider wrap keeps short labels on one
        // line instead of growing each node to two.
        flowchart: {
          curve: "basis",
          padding: 10,
          nodeSpacing: 32,
          rankSpacing: 28,
          wrappingWidth: 260,
        },
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
        ref={hostRef}
        data-testid="mermaid-diagram-pending"
        aria-busy
        className={cn(
          "h-40 animate-pulse rounded-lg border border-border bg-surface-raised/40",
          className,
        )}
      />
    );
  }

  return (
    <div
      ref={hostRef}
      data-testid="mermaid-diagram"
      // Cap the height so a tall diagram scales down (the SVG keeps its
      // aspect ratio) instead of filling the page.
      className={cn(
        "flex justify-center overflow-x-auto rounded-lg border border-border bg-surface p-4 [&_svg]:max-h-96",
        className,
      )}
      // The SVG comes from Mermaid in strict mode, which sanitizes it.
      dangerouslySetInnerHTML={{ __html: state.svg }}
    />
  );
}
