import { Typography } from "#/ui/typography";
import { ToolParameters } from "./tool-parameters";
import { ToggleButton } from "./toggle-button";
import { ChatCompletionToolParam } from "#/types/v1/core";

interface FunctionData {
  name?: string;
  description?: string;
  parameters?: Record<string, unknown>;
}

interface ToolData {
  type?: string;
  function?: FunctionData;
  name?: string;
  description?: string;
  parameters?: Record<string, unknown>;
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
  const name =
    functionData.name ||
    (toolData.type === "function" && toolData.function?.name) ||
    "";
  const description =
    functionData.description ||
    (toolData.type === "function" && toolData.function?.description) ||
    "";
  const parameters =
    functionData.parameters ||
    (toolData.type === "function" && toolData.function?.parameters) ||
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
          <div className="mt-2 mb-3">
            <Typography.Text className="text-sm whitespace-pre-wrap text-gray-300 leading-relaxed">
              {String(description)}
            </Typography.Text>
          </div>

          {/* Parameters section */}
          {parameters && <ToolParameters parameters={parameters} />}
        </div>
      )}
    </div>
  );
}
