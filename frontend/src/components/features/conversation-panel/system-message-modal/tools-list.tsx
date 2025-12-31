import { ChatCompletionToolParam } from "#/types/v1/core";
import { ToolItem } from "./tool-item";

interface ToolsListProps {
  tools: Array<Record<string, unknown>> | ChatCompletionToolParam[];
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
      {tools.map((tool, index) => (
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
