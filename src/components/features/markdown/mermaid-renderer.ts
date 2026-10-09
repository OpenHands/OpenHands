import DOMPurify from "dompurify";

/**
 * Mermaid render pipeline shared by the inline diagram card and the
 * `.mmd` / `.mermaid` artifact preview.
 *
 * The Mermaid library is heavy (d3, cytoscape, katex, …), so it is loaded
 * through a dynamic `import()` only when a diagram is actually rendered — a
 * conversation without a Mermaid block never downloads it. The module itself
 * has no static dependency on `mermaid`, so importing this file does not pull
 * the library into the eager bundle.
 */

/**
 * Mermaid's strictest rendering mode: it escapes HTML in labels, drops click
 * handlers and links, and sanitizes the output. The input is agent/model
 * output, so the `loose` default is never acceptable.
 */
export const MERMAID_SECURITY_LEVEL = "strict" as const;

type MermaidApi = {
  initialize: (config: Record<string, unknown>) => void;
  parse: (text: string) => Promise<unknown>;
  render: (id: string, text: string) => Promise<{ svg: string }>;
};

let mermaidPromise: Promise<MermaidApi> | null = null;

/** Loads and configures Mermaid exactly once per session. */
function loadMermaid(): Promise<MermaidApi> {
  if (!mermaidPromise) {
    mermaidPromise = import("mermaid").then((module) => {
      const mermaid = module.default as unknown as MermaidApi;
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: MERMAID_SECURITY_LEVEL,
        // Disable HTML labels so diagram text can never introduce an HTML
        // subtree (the strict level already escapes it, this is defense in
        // depth and also keeps the SVG self-contained).
        htmlLabels: false,
        fontFamily: "inherit",
      });
      return mermaid;
    });
  }
  return mermaidPromise;
}

/**
 * Runs Mermaid's own output through DOMPurify restricted to the SVG profile.
 * Mermaid already sanitizes in strict mode; this second pass guarantees the
 * string handed to `dangerouslySetInnerHTML` carries no `<script>`, no event
 * handler attributes, no `javascript:` URLs, and no `<foreignObject>`.
 */
export function sanitizeMermaidSvg(svg: string): string {
  return DOMPurify.sanitize(svg, {
    USE_PROFILES: { svg: true, svgFilters: true },
    FORBID_TAGS: ["script", "foreignObject"],
    FORBID_ATTR: ["onload", "onerror", "onclick", "onmouseover"],
  });
}

/**
 * Renders Mermaid `source` to a sanitized SVG string.
 *
 * Rejects (rather than returning partial output) when the diagram is invalid
 * or unsupported, so the caller can fall back to showing the raw source.
 */
export async function renderMermaidToSafeSvg(
  id: string,
  source: string,
): Promise<string> {
  const mermaid = await loadMermaid();
  await mermaid.parse(source);
  const { svg } = await mermaid.render(id, source);
  return sanitizeMermaidSvg(svg);
}

/** Test-only: clears the memoized loader so the dynamic-import boundary can
 *  be observed from a clean slate. */
export function resetMermaidLoaderForTests(): void {
  mermaidPromise = null;
}
