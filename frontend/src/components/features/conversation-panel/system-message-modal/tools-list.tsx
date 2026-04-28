import { ToolItem } from "./tool-item";
import { ChatCompletionToolParam } from "#/types/v1/core";

interface ToolsListProps {
  tools: Array<{
    tool: Record<string, unknown> | ChatCompletionToolParam;
    index: number;
  }>;
  expandedTools: Record<number, boolean>;
  onToggleTool: (index: number) => void;
}

export function ToolsList({
  tools,
  expandedTools,
  onToggleTool,
}: ToolsListProps) {
  return (
    <div className="p-2 space-y-3">
      {tools.map(({ tool, index }) => (
        <ToolItem
          key={index}
          tool={tool}
          index={index}
          isExpanded={expandedTools[index] || false}
          onToggle={onToggleTool}
        />
      ))}
    </div>
  );
}
