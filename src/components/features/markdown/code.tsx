import React from "react";
import { ExtraProps } from "react-markdown";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import { CopyableContentWrapper } from "#/components/shared/buttons/copyable-content-wrapper";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { openWorkspaceFile } from "#/services/canvas-ui";
import { looksLikeWorkspaceFilePath } from "#/utils/path-utils";
import { cn } from "#/utils/utils";
import { SyntaxHighlighter } from "./syntax-highlighter";

// See https://github.com/remarkjs/react-markdown?tab=readme-ov-file#use-custom-components-syntax-highlight

function MarkdownFilePathLink({
  path,
  className,
  children,
}: {
  path: string;
  className?: string;
  children: React.ReactNode;
}) {
  const { conversationId } = useOptionalConversationId();
  const { data: conversation } = useActiveConversation();

  return (
    <button
      type="button"
      data-testid="markdown-file-path-link"
      title={path}
      className={cn(
        className,
        "cursor-pointer rounded border border-surface-raised bg-surface-raised px-[0.4em] py-[0.2em] font-mono text-foreground hover:underline",
      )}
      onClick={(event) => {
        event.stopPropagation();
        openWorkspaceFile(
          path,
          conversationId ?? null,
          conversation?.workspace?.working_dir,
        );
      }}
    >
      {children}
    </button>
  );
}

/**
 * Component to render code blocks in markdown.
 */
export function code({
  children,
  className,
}: React.ClassAttributes<HTMLElement> &
  React.HTMLAttributes<HTMLElement> &
  ExtraProps) {
  const match = /language-(\w+)/.exec(className || ""); // get the language
  const codeString = String(children).replace(/\n$/, "");

  if (!match) {
    const isMultiline = String(children).includes("\n");

    if (!isMultiline) {
      if (looksLikeWorkspaceFilePath(codeString)) {
        return (
          <MarkdownFilePathLink path={codeString} className={className}>
            {children}
          </MarkdownFilePathLink>
        );
      }

      return (
        <code
          className={cn(
            className,
            "bg-surface-raised text-foreground border border-surface-raised rounded px-[0.4em] py-[0.2em]",
          )}
        >
          {children}
        </code>
      );
    }

    return (
      <CopyableContentWrapper text={codeString}>
        <pre className="bg-surface-raised text-foreground border border-surface-raised rounded p-[1em] overflow-auto">
          <code className={className}>{codeString}</code>
        </pre>
      </CopyableContentWrapper>
    );
  }

  return (
    <CopyableContentWrapper text={codeString}>
      <SyntaxHighlighter
        className="rounded-lg"
        style={vscDarkPlus}
        language={match?.[1]}
        PreTag="div"
      >
        {codeString}
      </SyntaxHighlighter>
    </CopyableContentWrapper>
  );
}
