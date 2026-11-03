import React from "react";
import { useColorTheme } from "#/hooks/use-color-theme";
import { COLOR_THEMES } from "#/themes/color-themes";

interface MermaidDiagramProps {
  /** The Mermaid source from a ```mermaid fenced code block. */
  source: string;
  /** Shown while the diagram renders, and instead of it when it is invalid. */
  fallback: React.ReactNode;
}

type RenderState =
  | { status: "pending" }
  | { status: "rendered"; svg: string }
  | { status: "invalid" };

let nextDiagramId = 0;

/**
 * Renders a ```mermaid fenced code block as a diagram.
 *
 * Mermaid is large, so it is imported on the first diagram rather than with
 * the app. The source is untrusted (chat replies, automation descriptions),
 * so it renders with `securityLevel: "strict"`: Mermaid sanitizes labels and
 * disables click handlers. Invalid source, including a streamed block that
 * is not finished yet, shows the fallback (the plain code block) instead.
 */
export function MermaidDiagram({ source, fallback }: MermaidDiagramProps) {
  const colorTheme = useColorTheme();
  const isDark = COLOR_THEMES[colorTheme]?.appearance === "dark";
  const [state, setState] = React.useState<RenderState>({ status: "pending" });

  React.useEffect(() => {
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
      if (!cancelled) setState({ status: "rendered", svg });
    })().catch(() => {
      if (!cancelled) setState({ status: "invalid" });
    });

    return () => {
      cancelled = true;
    };
  }, [source, isDark]);

  if (state.status !== "rendered") {
    return fallback;
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
