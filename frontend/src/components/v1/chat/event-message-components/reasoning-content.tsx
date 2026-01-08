/* eslint-disable i18next/no-literal-string */
import React, { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

import { ActionEvent } from "#/types/v1/core";
import {
  ThinkingBlock,
  RedactedThinkingBlock,
} from "#/types/v1/core/base/event";
import { MarkdownRenderer } from "../../../features/markdown/markdown-renderer";

interface ReasoningContentProps {
  event: ActionEvent;
}

export function ReasoningContent({ event }: ReasoningContentProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  // Check if there's any reasoning content to display
  const hasReasoningContent =
    event.reasoning_content && event.reasoning_content.trim().length > 0;
  const hasThinkingBlocks =
    event.thinking_blocks && event.thinking_blocks.length > 0;

  if (!hasReasoningContent && !hasThinkingBlocks) {
    return null;
  }

  const renderThinkingBlocks = () => {
    if (!hasThinkingBlocks) return null;

    return event.thinking_blocks.map(
      (block: ThinkingBlock | RedactedThinkingBlock, index) => {
        if (block.type === "thinking") {
          return (
            <div key={index} className="mb-2">
              <div className="text-xs text-gray-500 mb-1">
                Thinking Block {index + 1}
              </div>
              <div className="bg-gray-50 dark:bg-gray-800 rounded p-2 text-sm">
                <MarkdownRenderer includeStandard>
                  {block.thinking}
                </MarkdownRenderer>
              </div>
            </div>
          );
        }
        if (block.type === "redacted_thinking") {
          return (
            <div key={index} className="mb-2">
              <div className="text-xs text-gray-500 mb-1">
                Redacted Thinking Block {index + 1}
              </div>
              <div className="bg-gray-50 dark:bg-gray-800 rounded p-2 text-sm italic text-gray-600">
                [Redacted thinking content]
              </div>
            </div>
          );
        }
        return null;
      },
    );
  };

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
        <span>Reasoning</span>
      </button>

      {isExpanded && (
        <div className="mt-2 pl-4 border-l-2 border-gray-200 dark:border-gray-700">
          {hasReasoningContent && (
            <div className="mb-3">
              <div className="text-xs text-gray-500 mb-1">
                Reasoning Content
              </div>
              <div className="bg-gray-50 dark:bg-gray-800 rounded p-3 text-sm">
                <MarkdownRenderer includeStandard>
                  {event.reasoning_content || ""}
                </MarkdownRenderer>
              </div>
            </div>
          )}

          {hasThinkingBlocks && (
            <div>
              <div className="text-xs text-gray-500 mb-2">Thinking Blocks</div>
              {renderThinkingBlocks()}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
