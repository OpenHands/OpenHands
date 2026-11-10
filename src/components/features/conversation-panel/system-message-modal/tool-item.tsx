import { ToolParameters } from "./tool-parameters";
import { ToggleButton } from "./toggle-button";
import { ChatCompletionToolParam } from "#/types/agent-server/core";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { MarkdownRenderer } from "../../markdown/markdown-renderer";

interface FunctionData {
  name?: string;
  description?: string;
  parameters?: Record<string, unknown>;
}

interface ToolData {
  // OpenAI-compatible format
  type?: string;
  function?: FunctionData;
  name?: string;
  description?: string;
  parameters?: Record<string, unknown>;
  // agent-server format
  title?: string;
  kind?: string;
  annotations?: {
    title?: string;
  };
  source?: "conversation" | "global" | "builtin";
}

interface ToolItemProps {
  tool: Record<string, unknown> | ChatCompletionToolParam;
  index: number;
  isExpanded: boolean;
  onToggle: (index: number) => void;
}

export function ToolItem({ tool, index, isExpanded, onToggle }: ToolItemProps) {
  const { t } = useTranslation("openhands");
  // Extract function data from the nested structure
  const toolData = tool as ToolData;
  const functionData = toolData.function || toolData;

  // Extract tool name/title from supported tool schemas
  const name =
    // agent-server format: check for title field (root level or in annotations)
    toolData.title ||
    toolData.annotations?.title ||
    // OpenAI-compatible format: check for function.name or name
    functionData.name ||
    (toolData.type === "function" && toolData.function?.name) ||
    "";

  // Extract description from supported tool schemas
  const description =
    // agent-server format: description at root level
    toolData.description ||
    // OpenAI-compatible format: description in function object
    functionData.description ||
    (toolData.type === "function" && toolData.function?.description) ||
    "";

  // Extract parameters from supported tool schemas
  const parameters =
    // OpenAI-compatible format: parameters in function object
    functionData.parameters ||
    (toolData.type === "function" && toolData.function?.parameters) ||
    // agent-server format: parameters at root level (if present)
    toolData.parameters ||
    null;

  return (
    <div>
      <ToggleButton
        title={String(name)}
        isExpanded={isExpanded}
        onClick={() => onToggle(index)}
        badge={
          toolData.source === "conversation" || toolData.source === "global" ? (
            <span className="rounded border border-border px-2 py-0.5 text-xs font-normal text-muted">
              {t(
                toolData.source === "conversation"
                  ? I18nKey.SYSTEM_MESSAGE_MODAL$CONVERSATION_ONLY_TOOL
                  : I18nKey.SYSTEM_MESSAGE_MODAL$GLOBAL_TOOL,
              )}
            </span>
          ) : undefined
        }
      />

      {isExpanded && (
        <div className="px-3 pb-3 pt-1 border-t border-border">
          <div className="mt-2 mb-3 text-sm text-text-tertiary leading-relaxed">
            <MarkdownRenderer>{String(description)}</MarkdownRenderer>
          </div>

          {parameters && <ToolParameters parameters={parameters} />}
        </div>
      )}
    </div>
  );
}
