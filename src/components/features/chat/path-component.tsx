import { createContext, ReactNode, useContext } from "react";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { openWorkspaceFile } from "#/services/canvas-ui";
import EventLogger from "#/utils/event-logger";

/**
 * When false, PathComponent renders a non-interactive span. EventGroup's
 * header wraps titles in a toggle <button>, so nested path buttons would be
 * invalid markup — that context sets this to false.
 */
export const PathInteractiveContext = createContext(true);

/**
 * Decodes HTML entities in a string
 * @param text The text to decode
 * @returns The decoded text
 */
const decodeHtmlEntities = (text: string): string => {
  const textarea = document.createElement("textarea");
  textarea.innerHTML = text;
  return textarea.value;
};

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
 * Displays only the filename, with the full path on hover.
 * Click opens the Files drawer on that path (when interactive).
 */
function PathComponent(props: { children?: ReactNode }) {
  const { children } = props;
  const { conversationId } = useOptionalConversationId();
  const interactive = useContext(PathInteractiveContext);

  const processPath = (path: string) => {
    try {
      const decodedPath = decodeHtmlEntities(path);
      const filename = extractFilename(decodedPath);
      if (!interactive) {
        return (
          <span
            className="font-mono font-normal tracking-tight"
            title={decodedPath}
          >
            {filename}
          </span>
        );
      }
      return (
        <button
          type="button"
          data-testid="path-component-link"
          className="cursor-pointer font-mono font-normal tracking-tight hover:underline"
          title={decodedPath}
          onClick={(event) => {
            event.stopPropagation();
            openWorkspaceFile(decodedPath, conversationId);
          }}
        >
          {filename}
        </button>
      );
    } catch (e) {
      EventLogger.error(String(e));
      return (
        <span className="font-mono font-normal tracking-tight">{path}</span>
      );
    }
  };

  if (Array.isArray(children)) {
    const processedChildren = children.map((child, index) =>
      typeof child === "string" ? (
        <span key={`${child}-${index}`}>{processPath(child)}</span>
      ) : (
        child
      ),
    );

    return (
      <span className="font-normal tracking-tight">{processedChildren}</span>
    );
  }

  if (typeof children === "string") {
    return (
      <span className="font-normal tracking-tight">
        {processPath(children)}
      </span>
    );
  }

  return (
    <span className="font-mono font-normal tracking-tight">{children}</span>
  );
}

export { PathComponent, isLikelyDirectory };
