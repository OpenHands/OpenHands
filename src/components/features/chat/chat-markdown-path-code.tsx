import React, { createContext, useContext } from "react";
import type { ExtraProps } from "react-markdown";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { useNavigation } from "#/context/navigation-context";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useWorkspaceFiles } from "#/hooks/query/use-workspace-files";
import { openWorkspaceFile } from "#/services/canvas-ui";
import { looksLikeWorkspaceFilePath, toFilesTabPath } from "#/utils/path-utils";
import { cn } from "#/utils/utils";
import { code as defaultCode } from "../markdown/code";
import { anchor as defaultAnchor } from "../markdown/anchor";

/**
 * href scheme the Agent Server emits for conversation mentions
 * (`conversation:<32-hex id>`; dashed UUIDs accepted leniently).
 */
export const CONVERSATION_MENTION_PROTOCOL = "conversation";

/** href protocols chat bubbles pass through markdown sanitizing. */
export const CHAT_BUBBLE_HREF_PROTOCOLS = [
  CONVERSATION_MENTION_PROTOCOL,
] as const;

const CONVERSATION_MENTION_HREF_PATTERN =
  /^conversation:([0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$/i;

/**
 * Parse a `conversation:` mention href into the conversation id in
 * dashed-UUID lowercase form, or null when the id is neither a 32-hex id
 * nor a dashed UUID (such a mention renders as plain text, not a dead
 * link).
 */
export function parseConversationMentionHref(href: string): string | null {
  const match = CONVERSATION_MENTION_HREF_PATTERN.exec(href);
  if (!match) return null;
  const id = match[1].toLowerCase();
  if (id.length === 32) {
    return [
      id.slice(0, 8),
      id.slice(8, 12),
      id.slice(12, 16),
      id.slice(16, 20),
      id.slice(20, 32),
    ].join("-");
  }
  return id;
}

/** True while rendering Markdown descendants of an existing `<a>`. */
const InMarkdownLinkContext = createContext(false);

/**
 * Workspace paths available for chat Markdown linking. Absent outside
 * ChatInterface (e.g. unit tests) so ChatCode never requires QueryClient.
 */
export const WorkspaceFilesForChatContext = createContext<string[] | undefined>(
  undefined,
);

/** Fetches workspace files once for the chat tree and exposes them to path links. */
export function WorkspaceFilesForChatProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const { data } = useWorkspaceFiles();
  return (
    <WorkspaceFilesForChatContext.Provider value={data}>
      {children}
    </WorkspaceFilesForChatContext.Provider>
  );
}

type CodeProps = React.ClassAttributes<HTMLElement> &
  React.HTMLAttributes<HTMLElement> &
  ExtraProps;

type AnchorProps = React.ClassAttributes<HTMLAnchorElement> &
  React.AnchorHTMLAttributes<HTMLAnchorElement> &
  ExtraProps;

type StrongProps = React.ClassAttributes<HTMLElement> &
  React.HTMLAttributes<HTMLElement> &
  ExtraProps;

/** Flatten React children to plain text, or null if mixed nodes. */
function getPlainText(children: React.ReactNode): string | null {
  if (children == null || typeof children === "boolean") return "";
  if (typeof children === "string" || typeof children === "number") {
    return String(children);
  }
  if (Array.isArray(children)) {
    const parts = children.map(getPlainText);
    if (parts.some((part) => part === null)) return null;
    return parts.join("");
  }
  return null;
}

/**
 * Only link paths that currently exist in the conversation workspace.
 * workingDir comes from ConversationService — same source as navigate_to_file.
 */
function useExistingWorkspacePath(candidate: string): string | null {
  const files = useContext(WorkspaceFilesForChatContext);
  if (!files?.length || !looksLikeWorkspaceFilePath(candidate)) return null;

  const workingDir =
    ConversationService.getCurrentConversation()?.workspace?.working_dir;
  const normalized = toFilesTabPath(candidate, workingDir);
  if (!normalized) return null;
  return files.includes(normalized) ? normalized : null;
}

function ChatMarkdownPathLink({
  path,
  className,
  children,
}: {
  path: string;
  className?: string;
  children: React.ReactNode;
}) {
  const { conversationId } = useOptionalConversationId();

  return (
    <button
      type="button"
      data-testid="markdown-file-path-link"
      title={path}
      className={cn(
        className,
        "cursor-pointer rounded border border-surface-raised bg-surface-raised px-code-inline-x py-code-inline-y font-mono text-foreground hover:underline",
      )}
      onClick={(event) => {
        event.stopPropagation();
        openWorkspaceFile(path, conversationId);
      }}
    >
      {children}
    </button>
  );
}

/**
 * Chat-only inline code: existing workspace paths become Files-drawer buttons.
 * Nested under an existing Markdown link → stay plain `<code>`.
 */
export function ChatCode(props: CodeProps) {
  const { children, className } = props;
  const inLink = useContext(InMarkdownLinkContext);
  const match = /language-(\w+)/.exec(className || "");
  const codeString = String(children).replace(/\n$/, "");
  const isMultiline = String(children).includes("\n");
  const existingPath = useExistingWorkspacePath(
    !match && !isMultiline && !inLink ? codeString : "",
  );

  if (existingPath) {
    return (
      <ChatMarkdownPathLink path={existingPath} className={className}>
        {children}
      </ChatMarkdownPathLink>
    );
  }

  return defaultCode(props);
}

/**
 * Agents often emphasize paths with `**file.md**` instead of backticks.
 * Link those only when the whole strong span is an existing workspace file.
 */
export function ChatStrong(props: StrongProps) {
  const { children } = props;
  const inLink = useContext(InMarkdownLinkContext);
  const text = getPlainText(children)?.trim() ?? "";
  const existingPath = useExistingWorkspacePath(!inLink ? text : "");

  if (existingPath) {
    return (
      <ChatMarkdownPathLink path={existingPath}>
        {children}
      </ChatMarkdownPathLink>
    );
  }

  return <strong>{children}</strong>;
}

/** Marks descendants so nested path tokens are not turned into a button. */
export function ChatAnchor(props: AnchorProps) {
  const { href, children } = props;
  const { navigate } = useNavigation();

  // A mention link opens the referenced conversation in-app instead of
  // delegating to the browser (which cannot handle the scheme).
  if (typeof href === "string") {
    const mentionedId = parseConversationMentionHref(href);
    if (mentionedId) {
      return (
        <button
          type="button"
          data-testid="conversation-mention-link"
          title={href}
          className="cursor-pointer rounded border border-surface-raised bg-surface-raised px-code-inline-x py-code-inline-y text-foreground hover:underline"
          onClick={(event) => {
            event.stopPropagation();
            navigate(`/conversations/${mentionedId}`);
          }}
        >
          <InMarkdownLinkContext.Provider value>
            {children}
          </InMarkdownLinkContext.Provider>
        </button>
      );
    }

    // A conversation link whose id is not a valid UUID: keep the title as
    // plain text instead of a dead empty-href anchor.
    if (href.toLowerCase().startsWith(`${CONVERSATION_MENTION_PROTOCOL}:`)) {
      return (
        <span data-testid="conversation-mention-invalid">
          <InMarkdownLinkContext.Provider value>
            {children}
          </InMarkdownLinkContext.Provider>
        </span>
      );
    }
  }

  return (
    <InMarkdownLinkContext.Provider value>
      {defaultAnchor(props)}
    </InMarkdownLinkContext.Provider>
  );
}
