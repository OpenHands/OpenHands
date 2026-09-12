import React from "react";
import { ExtraProps } from "react-markdown";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import { CopyableContentWrapper } from "#/components/shared/buttons/copyable-content-wrapper";
import { cn } from "#/utils/utils";
import { SyntaxHighlighter } from "./syntax-highlighter";

// See https://github.com/remarkjs/react-markdown?tab=readme-ov-file#use-custom-components-syntax-highlight

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
      return (
        <code
          dir="ltr"
          className={cn(
            className,
            "bg-surface-raised text-foreground border border-surface-raised rounded px-[0.4em] py-[0.2em] text-left",
          )}
        >
          {children}
        </code>
      );
    }

    return (
      <CopyableContentWrapper text={codeString}>
        <pre
          dir="ltr"
          className="bg-surface-raised text-foreground border border-surface-raised rounded p-[1em] overflow-auto text-left"
        >
          <code className={className}>{codeString}</code>
        </pre>
      </CopyableContentWrapper>
    );
  }

  return (
    <CopyableContentWrapper text={codeString}>
      <div dir="ltr" className="text-left">
        <SyntaxHighlighter
          className="rounded-lg"
          style={vscDarkPlus}
          language={match?.[1]}
          PreTag="div"
        >
          {codeString}
        </SyntaxHighlighter>
      </div>
    </CopyableContentWrapper>
  );
}
