import React from "react";
import { useTranslation } from "react-i18next";
import { SyntaxHighlighter } from "../../../markdown/syntax-highlighter";
import { CopyableContentWrapper } from "#/components/shared/buttons/copyable-content-wrapper";
import { MAX_CONTENT_LENGTH } from "#/components/conversation-events/chat/event-content-helpers/shared";
import { useColorTheme } from "#/hooks/use-color-theme";
import { I18nKey } from "#/i18n/declaration";
import { getSyntaxHighlighterTheme } from "#/themes/syntax-highlighter-themes";
import { useEventExpansionKey } from "#/components/features/chat/row-expansion-context";
import { usePersistentExpansion } from "#/stores/message-expansion-store";

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
  /**
   * Distinguishes this block's inline expand control from a sibling block in
   * the same event card when its state is persisted across virtualized
   * remounts. Defaults to `"code-block"`.
   */
  expansionId?: string;
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
  expansionId = "code-block",
}: CodeBlockProps) {
  const { t } = useTranslation("openhands");
  const colorTheme = useColorTheme();
  const expansionKey = useEventExpansionKey(expansionId);
  const [isExpanded, toggleExpanded] = usePersistentExpansion(
    expansionKey,
    false,
  );
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
      style={getSyntaxHighlighterTheme(colorTheme)}
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
          onClick={() => toggleExpanded()}
          className="self-start text-xs text-muted transition-colors hover:text-contrast hover:underline"
        >
          {toggleLabel}
        </button>
      )}
    </div>
  );
}
