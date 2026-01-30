import { ToolParameters } from "./tool-parameters";
import { ToggleButton } from "./toggle-button";
import { ChatCompletionToolParam } from "#/types/v1/core";
import { MarkdownRenderer } from "../../markdown/markdown-renderer";

interface FunctionData {
  name?: string;
  description?: string;
  parameters?: Record<string, unknown>;
}

interface ToolData {
  // V0/OpenAI format
  type?: string;
  function?: FunctionData;
  name?: string;
  description?: string;
  parameters?: Record<string, unknown>;
  // V1 format
  title?: string;
  kind?: string;
  annotations?: {
    title?: string;
  };
}

interface ToolItemProps {
  tool: Record<string, unknown> | ChatCompletionToolParam;
  index: number;
  isExpanded: boolean;
  onToggle: (index: number) => void;
}

export function ToolItem({ tool, index, isExpanded, onToggle }: ToolItemProps) {
  // Extract function data from the nested structure
  const toolData = tool as ToolData;
  const functionData = toolData.function || toolData;

  // Extract tool name/title - support both V0 and V1 formats
  const name =
    // V1 format: check for title field (root level or in annotations)
    toolData.title ||
    toolData.annotations?.title ||
    // V0 format: check for function.name or name
    functionData.name ||
    (toolData.type === "function" && toolData.function?.name) ||
    "";

  // Extract description - support both V0 and V1 formats
  const description =
    // V1 format: description at root level
    toolData.description ||
    // V0 format: description in function object
    functionData.description ||
    (toolData.type === "function" && toolData.function?.description) ||
    "";

  // Extract parameters - support both V0 and V1 formats
  const parameters =
    // V0 format: parameters in function object
    functionData.parameters ||
    (toolData.type === "function" && toolData.function?.parameters) ||
    // V1 format: parameters at root level (if present)
    toolData.parameters ||
    null;

  return (
    <div className="rounded-md overflow-hidden">
      <ToggleButton
        title={String(name)}
        isExpanded={isExpanded}
        onClick={() => onToggle(index)}
      />

      {isExpanded && (
        <div className="px-2 pb-3 pt-1">
          <div className="mt-2 mb-3 text-sm text-gray-300 leading-relaxed">
            <MarkdownRenderer>{String(description)}</MarkdownRenderer>
          </div>

          {/* Parameters section */}
          {parameters && <ToolParameters parameters={parameters} />}
        </div>
      )}
    </div>
  );
}
