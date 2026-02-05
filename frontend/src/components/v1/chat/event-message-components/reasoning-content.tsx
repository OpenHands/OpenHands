import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { ChevronDown, ChevronRight } from "lucide-react";

import { I18nKey } from "#/i18n/declaration";
import { ActionEvent } from "#/types/v1/core";
import {
  ThinkingBlock,
  RedactedThinkingBlock,
} from "#/types/v1/core/base/event";
import { Typography } from "#/ui/typography";
import { MarkdownRenderer } from "../../../features/markdown/markdown-renderer";

interface ReasoningContentProps {
  event: ActionEvent;
}

interface ThinkingBlockItemProps {
  block: ThinkingBlock;
  index: number;
}

interface RedactedThinkingBlockItemProps {
  index: number;
}

interface ThinkingBlocksListProps {
  blocks: (ThinkingBlock | RedactedThinkingBlock)[];
}

interface ReasoningContentSectionProps {
  content: string;
}

function ThinkingBlockItem({ block, index }: ThinkingBlockItemProps) {
  const { t } = useTranslation();
  return (
    <div className="mb-2">
      <div className="text-xs text-gray-500 mb-1">
        {t(I18nKey.REASONING_CONTENT$THINKING_BLOCK_INDEX, {
          index: index + 1,
        })}
      </div>
      <div className="bg-gray-50 dark:bg-gray-800 rounded p-2 text-sm">
        <MarkdownRenderer includeStandard>{block.thinking}</MarkdownRenderer>
      </div>
    </div>
  );
}

function RedactedThinkingBlockItem({ index }: RedactedThinkingBlockItemProps) {
  const { t } = useTranslation();
  return (
    <div className="mb-2">
      <div className="text-xs text-gray-500 mb-1">
        {t(I18nKey.REASONING_CONTENT$REDACTED_THINKING_BLOCK_INDEX, {
          index: index + 1,
        })}
      </div>
      <div className="bg-gray-50 dark:bg-gray-800 rounded p-2 text-sm italic text-gray-600">
        {t(I18nKey.REASONING_CONTENT$REDACTED_PLACEHOLDER)}
      </div>
    </div>
  );
}

function ThinkingBlocksList({ blocks }: ThinkingBlocksListProps) {
  return (
    <>
      {blocks.map((block, index) => {
        if (block.type === "thinking") {
          return <ThinkingBlockItem key={index} block={block} index={index} />;
        }
        if (block.type === "redacted_thinking") {
          return <RedactedThinkingBlockItem key={index} index={index} />;
        }
        return null;
      })}
    </>
  );
}

function ReasoningContentSection({ content }: ReasoningContentSectionProps) {
  const { t } = useTranslation();
  return (
    <div className="mb-3">
      <div className="text-xs text-gray-500 mb-1">
        {t(I18nKey.REASONING_CONTENT$REASONING_CONTENT_LABEL)}
      </div>
      <div className="bg-gray-50 dark:bg-gray-800 rounded p-3 text-sm">
        <MarkdownRenderer includeStandard>{content}</MarkdownRenderer>
      </div>
    </div>
  );
}

export function ReasoningContent({ event }: ReasoningContentProps) {
  const { t } = useTranslation();
  const [isExpanded, setIsExpanded] = useState(false);

  // Check if there's any reasoning content to display
  const hasReasoningContent =
    event.reasoning_content && event.reasoning_content.trim().length > 0;
  const hasThinkingBlocks =
    event.thinking_blocks && event.thinking_blocks.length > 0;

  if (!hasReasoningContent && !hasThinkingBlocks) {
    return null;
  }

  return (
    <div className="mb-2">
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-1 text-xs text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 transition-colors"
      >
        {isExpanded ? (
          <ChevronDown className="h-3 w-3" />
        ) : (
          <ChevronRight className="h-3 w-3" />
        )}
        <Typography.Text className="text-xs text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200">
          {t(I18nKey.REASONING_CONTENT$REASONING)}
        </Typography.Text>
      </button>

      {isExpanded && (
        <div className="mt-2 pl-4 border-l-2 border-gray-200 dark:border-gray-700">
          {hasReasoningContent && (
            <ReasoningContentSection content={event.reasoning_content || ""} />
          )}

          {hasThinkingBlocks && (
            <div>
              <div className="text-xs text-gray-500 mb-2">
                {t(I18nKey.REASONING_CONTENT$THINKING_BLOCKS)}
              </div>
              <ThinkingBlocksList blocks={event.thinking_blocks} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
