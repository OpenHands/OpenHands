import React from "react";
import { useTranslation } from "react-i18next";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import { SyntaxHighlighter } from "../../../markdown/syntax-highlighter";
import { CopyableContentWrapper } from "#/components/shared/buttons/copyable-content-wrapper";
import { MAX_CONTENT_LENGTH } from "#/components/conversation-events/chat/event-content-helpers/shared";
import { I18nKey } from "#/i18n/declaration";

interface CodeBlockProps {
  code: string;
  /** Prism language hint (e.g. "bash", "python"). */
  language?: string;
  /** Show a copy button on hover. Defaults to true. */
  copy?: boolean;
  /** Text shown when code is empty. */
  placeholder?: string;
  /** Let truncated content be expanded inline. Defaults to false. */
  expandable?: boolean;
  /** Wrap long lines instead of horizontal-only scrolling. Defaults to false. */
  wrapLongLines?: boolean;
}

/**
 * Syntax-highlighted code block with an optional hover copy button. Long
 * content is truncated to the same limit the markdown path uses, optionally
 * with an inline expand control. The copy button always yields the full,
 * untruncated text.
 */
export function CodeBlock({
  code,
  language,
  copy = true,
  placeholder,
  expandable = false,
  wrapLongLines = false,
}: CodeBlockProps) {
  const { t } = useTranslation("openhands");
  const [isExpanded, setIsExpanded] = React.useState(false);
  const isTruncated = code.length > MAX_CONTENT_LENGTH;
  const display =
    isTruncated && !(expandable && isExpanded)
      ? `${code.slice(0, MAX_CONTENT_LENGTH)}…`
      : code;
  const text = display.trim() || placeholder || "";
  const canCopy = copy && code.trim().length > 0;
  const toggleLabel = isExpanded
    ? t(I18nKey.BUTTON$COLLAPSE)
    : t(I18nKey.BUTTON$EXPAND);

  const block = (
    <SyntaxHighlighter
      className="rounded-lg text-xs"
      style={vscDarkPlus}
      language={language}
      PreTag="div"
      wrapLongLines={wrapLongLines}
      customStyle={wrapLongLines ? { whiteSpace: "pre-wrap" } : undefined}
      codeTagProps={
        wrapLongLines ? { style: { whiteSpace: "pre-wrap" } } : undefined
      }
    >
      {text}
    </SyntaxHighlighter>
  );

  return (
    <div className="flex flex-col gap-1">
      {canCopy ? (
        <CopyableContentWrapper text={code}>{block}</CopyableContentWrapper>
      ) : (
        block
      )}
      {expandable && isTruncated && (
        <button
          type="button"
          onClick={() => setIsExpanded((prev) => !prev)}
          className="self-start text-xs text-muted transition-colors hover:text-white hover:underline"
        >
          {toggleLabel}
        </button>
      )}
    </div>
  );
}
