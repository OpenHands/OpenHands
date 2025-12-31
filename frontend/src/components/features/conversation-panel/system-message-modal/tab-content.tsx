import { SystemMessageContent } from "./system-message-content";
import { ToolsList } from "./tools-list";
import { EmptyToolsState } from "./empty-tools-state";
import { ChatCompletionToolParam } from "#/types/v1/core";

interface TabContentProps {
  activeTab: "system" | "tools";
  systemMessage: {
    content: string;
    tools: Array<Record<string, unknown>> | ChatCompletionToolParam[] | null;
  };
  expandedTools: Record<number, boolean>;
  onToggleTool: (index: number) => void;
}

export function TabContent({
  activeTab,
  systemMessage,
  expandedTools,
  onToggleTool,
}: TabContentProps) {
  if (activeTab === "system") {
    return <SystemMessageContent content={systemMessage.content} />;
  }

  if (activeTab === "tools") {
    if (systemMessage.tools && systemMessage.tools.length > 0) {
      return (
        <ToolsList
          tools={systemMessage.tools}
          expandedTools={expandedTools}
          onToggleTool={onToggleTool}
        />
      );
    }

    return <EmptyToolsState />;
  }

  return null;
}
