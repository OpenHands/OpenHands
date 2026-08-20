import { useQuery } from "@tanstack/react-query";

import { readCloudConversationFile } from "#/api/cloud/conversation-service.api";
import { getActiveBackend } from "#/api/backend-registry/active-store";
import { getGitPath } from "#/utils/get-git-path";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";
import {
  joinWorkspaceUrl,
  useWorkspaceSession,
} from "#/hooks/query/use-workspace-session";
import { useWorkspaceMutationCounter } from "#/stores/use-workspace-mutation-counter";

// Magic-number sniff for common binary formats we can render via iframe.
const IMAGE_EXTENSIONS = new Set([
  "png",
  "jpg",
  "jpeg",
  "gif",
  "webp",
  "bmp",
  "ico",
  "svg",
  "avif",
]);

const PDF_EXTENSIONS = new Set(["pdf"]);

export type WorkspaceFileKind = "text" | "image" | "pdf" | "binary";

export interface WorkspaceFileContent {
  path: string;
  kind: WorkspaceFileKind;
  /** Decoded text contents — only populated when kind === "text". */
  text: string | null;
  /**
   * URL pointing at the file on the agent server's static workspace
   * fileserver (the `/api/conversations/{id}/workspace/...` route minted
   * by `RemoteWorkspace.startWorkspaceSession`). Suitable to use as an
   * `<iframe src>` or `<img src>` — the workspace-session cookie
   * authenticates the browser request, and relative asset references
   * inside an HTML preview resolve naturally against this URL.
   */
  staticUrl: string;
  /** MIME type guessed from the file extension. */
  mimeType: string;
}

function getExtension(path: string): string {
  const idx = path.lastIndexOf(".");
  if (idx === -1) return "";
  return path.slice(idx + 1).toLowerCase();
}

function guessMimeType(path: string): string {
  const ext = getExtension(path);
  switch (ext) {
    case "html":
    case "htm":
      return "text/html";
    case "css":
      return "text/css";
    case "js":
    case "mjs":
    case "cjs":
      return "text/javascript";
    case "json":
      return "application/json";
    case "md":
    case "markdown":
      return "text/markdown";
    case "svg":
      return "image/svg+xml";
    case "png":
      return "image/png";
    case "jpg":
    case "jpeg":
      return "image/jpeg";
    case "gif":
      return "image/gif";
    case "webp":
      return "image/webp";
    case "bmp":
      return "image/bmp";
    case "ico":
      return "image/x-icon";
    case "avif":
      return "image/avif";
    case "pdf":
      return "application/pdf";
    default:
      return "text/plain";
  }
}

function classifyKind(path: string): WorkspaceFileKind {
  const ext = getExtension(path);
  if (IMAGE_EXTENSIONS.has(ext)) return "image";
  if (PDF_EXTENSIONS.has(ext)) return "pdf";
  // Everything else is treated as text and decoded; if decoding produces
  // null bytes we fall back to "binary" downstream.
  return "text";
}

function isLikelyBinary(buffer: ArrayBuffer): boolean {
  // Same heuristic git uses: presence of a NUL byte in the first ~8KB.
  const view = new Uint8Array(buffer, 0, Math.min(buffer.byteLength, 8000));
  for (let i = 0; i < view.length; i += 1) {
    if (view[i] === 0) return true;
  }
  return false;
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  // Chunk to stay under the call-stack limit of `String.fromCharCode(...arr)`
  // for larger files (>~100KB) while avoiding per-byte allocation.
  const CHUNK = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode.apply(
      null,
      bytes.subarray(i, i + CHUNK) as unknown as number[],
    );
  }
  return btoa(binary);
}

/**
 * Reads a single file out of the active conversation's workspace via the
 * agent server's static workspace fileserver and classifies it as
 * text/image/pdf/binary so the UI can pick a renderer.
 *
 * Image and PDF kinds are rendered directly from `staticUrl` (no fetch
 * here). Text/binary classification still requires reading the body so
 * we can run a NUL-byte sniff and decode UTF-8 for the plain/markdown
 * renderers.
 *
 * Pass a falsy `relativePath` to disable the query (e.g. when no file is
 * selected yet).
 */
export function useWorkspaceFileContent(relativePath: string | null) {
  const { data: conversation } = useActiveConversation();
  const runtimeIsReady = useRuntimeIsReady();
  const { data: workspaceSession } = useWorkspaceSession();
  // Bump on every agent-side file mutation so the query refetches the
  // currently-selected file's body even when the *path* hasn't changed.
  // The iframe / <img> cache-busting for the rich preview is handled at
  // the consumer (FileContentViewer / files-tab) by appending the same
  // counter to the staticUrl, so a single tick refreshes both the
  // decoded text and the iframe-rendered HTML's sibling assets.
  const workspaceMutationCount = useWorkspaceMutationCounter(
    (state) => state.count,
  );

  const conversationId = conversation?.id;
  const conversationUrl = conversation?.conversation_url;
  const sessionApiKey = conversation?.session_api_key;
  const selectedRepository = conversation?.selected_repository;
  const workingDir = conversation?.workspace?.working_dir?.trim();
  const baseUrl = workspaceSession?.baseUrl;
  const isCloud = getActiveBackend().backend.kind === "cloud";

  // The cloud `/file` endpoint downloads via the runtime's
  // `/api/file/download`, which rejects relative paths (400 → the cloud API
  // swallows it and returns ""). Anchor the file against the working dir the
  // same way the diff view builds its git-diff path (see use-unified-git-diff),
  // then force a leading slash since `getGitPath`'s default is relative.
  const gitPath = getGitPath(selectedRepository, workingDir);
  const workspaceRoot = gitPath.startsWith("/") ? gitPath : `/${gitPath}`;
  const absoluteFilePath = relativePath
    ? `${workspaceRoot}/${relativePath}`
    : null;

  return useQuery<WorkspaceFileContent>({
    queryKey: [
      "workspace-file-content",
      conversationId,
      conversationUrl,
      sessionApiKey,
      isCloud ? "cloud" : baseUrl,
      relativePath,
      absoluteFilePath,
      workspaceMutationCount,
    ],
    queryFn: async () => {
      if (!relativePath) throw new Error("No path");

      const kind = classifyKind(relativePath);
      const mimeType = guessMimeType(relativePath);

      if (isCloud) {
        // Cloud: fetch through the cloud API's first-class runtime proxy
        // (GET /api/v1/app-conversations/{id}/file), which avoids the
        // removed /api/cloud-proxy hop. The endpoint returns file content as
        // a string; binary files are detected via NUL-byte sniff on the
        // decoded result and served as base64 data URIs.
        const content = await readCloudConversationFile(
          conversationId!,
          absoluteFilePath!,
        );

        if (kind === "text") {
          // NUL-byte sniff on the decoded text to catch binary files that
          // the cloud endpoint decoded as UTF-8 (fallible but sufficient).
          const buf = new TextEncoder().encode(content);
          if (isLikelyBinary(buf.buffer)) {
            return {
              path: relativePath,
              kind: "binary",
              text: null,
              staticUrl: `data:application/octet-stream;base64,${arrayBufferToBase64(buf.buffer)}`,
              mimeType: "application/octet-stream",
            };
          }
          return {
            path: relativePath,
            kind: "text",
            text: content,
            staticUrl: `data:${mimeType};charset=utf-8;base64,${arrayBufferToBase64(buf.buffer)}`,
            mimeType,
          };
        }
        // Image / PDF via cloud API: the endpoint returns text, so binary
        // bytes are decoded as UTF-8 server-side and can't round-trip
        // faithfully. Best-effort base64 of the returned string — a proper
        // binary path needs a cloud download endpoint (the old byte-accurate
        // downloadFile route went through the removed /api/cloud-proxy).
        const buf = new TextEncoder().encode(content);
        return {
          path: relativePath,
          kind,
          text: null,
          staticUrl: `data:${mimeType};base64,${arrayBufferToBase64(buf.buffer)}`,
          mimeType,
        };
      }

      // Local: rely on the workspace-session cookie minted by
      // useWorkspaceSession to authenticate the same-origin static
      // fileserver fetch.
      if (!baseUrl) throw new Error("No workspace session");

      const staticUrl = joinWorkspaceUrl(baseUrl, relativePath);

      // Image / PDF: don't fetch the bytes — the consumer renders them
      // directly via `staticUrl` in an iframe or <img>. The browser
      // will attach the `oh_workspace_session_key` cookie minted by
      // `useWorkspaceSession` so the request authenticates without us
      // having to set any headers (which a top-level <iframe src> can't
      // do anyway).
      if (kind !== "text") {
        return {
          path: relativePath,
          kind,
          text: null,
          staticUrl,
          mimeType,
        };
      }

      // For our own fetch we also rely on the workspace-session cookie
      // (it travels because we opt in to credentialed requests). This
      // matches the auth path the iframe / <img> uses, and avoids a CORS
      // preflight for a custom header.
      const response = await fetch(staticUrl, {
        credentials: "include",
      });
      if (!response.ok) {
        throw new Error(`Failed to read ${relativePath}: ${response.status}`);
      }

      const buffer = await response.arrayBuffer();
      if (isLikelyBinary(buffer)) {
        return {
          path: relativePath,
          kind: "binary",
          text: null,
          staticUrl,
          mimeType: "application/octet-stream",
        };
      }

      const text = new TextDecoder("utf-8", { fatal: false }).decode(buffer);
      return {
        path: relativePath,
        kind: "text",
        text,
        staticUrl,
        mimeType,
      };
    },
    enabled:
      runtimeIsReady &&
      !!conversationId &&
      !!relativePath &&
      (isCloud || !!baseUrl),
    retry: false,
    staleTime: 1000 * 5,
    gcTime: 1000 * 60,
    meta: { disableToast: true },
  });
}
