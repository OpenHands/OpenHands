import { ReactNode } from "react";

/**
 * Checks if a path is likely a directory
 * @param path The full path
 * @returns True if the path is likely a directory
 */
const isLikelyDirectory = (path: string): boolean => {
  if (!path) return false;
  // Check if path already ends with a slash
  if (path.endsWith("/") || path.endsWith("\\")) return true;
  // Check if path has no extension (simple heuristic)
  const lastPart = path.split(/[/\\]/).pop() || "";
  // If the last part has no dots, it's likely a directory
  return !lastPart.includes(".");
};

/**
 * Extracts the filename from a path
 * @param path The full path
 * @returns The filename (last part of the path)
 */
const extractFilename = (path: string): string => {
  if (!path) return "";
  // Handle both Unix and Windows paths
  const parts = path.split(/[/\\]/);
  const filename = parts[parts.length - 1];

  // Add trailing slash for directories
  if (isLikelyDirectory(path) && !filename.endsWith("/")) {
    return `${filename}/`;
  }

  return filename;
};

/**
 * ``<Trans>`` component override for the ``<path>`` tag in translation
 * strings: displays the filename in the visible text and the full path
 * on hover (via the ``title`` attribute).
 *
 * Receives children verbatim — i18next no longer double-escapes
 * interpolated values (see ``i18n/index.ts``'s ``escapeValue: false``),
 * so the prior ``decodeHtmlEntities`` step had nothing to undo.
 */
function PathComponent(props: { children?: ReactNode }) {
  const { children } = props;

  const renderPath = (path: string) => (
    <span className="font-mono" title={path}>
      {extractFilename(path)}
    </span>
  );

  if (Array.isArray(children)) {
    const processedChildren = children.map((child) =>
      typeof child === "string" ? renderPath(child) : child,
    );
    return <strong className="font-mono">{processedChildren}</strong>;
  }

  if (typeof children === "string") {
    return <strong>{renderPath(children)}</strong>;
  }

  return <strong className="font-mono">{children}</strong>;
}

export { PathComponent, isLikelyDirectory };
