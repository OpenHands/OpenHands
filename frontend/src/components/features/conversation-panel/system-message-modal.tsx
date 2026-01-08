import { useState } from "react";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { SystemMessageHeader } from "./system-message-modal/system-message-header";
import { TabNavigation } from "./system-message-modal/tab-navigation";
import { TabContent } from "./system-message-modal/tab-content";
import { SystemMessageForModal } from "#/utils/system-message-adapter";

interface SystemMessageModalProps {
  systemMessage: SystemMessageForModal | null;
}

export function SystemMessageModal({ systemMessage }: SystemMessageModalProps) {
  const [activeTab, setActiveTab] = useState<"system" | "tools">("system");
  const [expandedTools, setExpandedTools] = useState<Record<number, boolean>>(
    {},
  );

  if (!systemMessage) {
    return null;
  }

  const toggleTool = (index: number) => {
    setExpandedTools((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  return (
    <ModalBody
      width="medium"
      className="max-h-[80vh] flex flex-col items-start"
    >
      <SystemMessageHeader
        agentClass={systemMessage.agent_class}
        openhandsVersion={systemMessage.openhands_version}
      />

      <div className="w-full">
        <TabNavigation
          activeTab={activeTab}
          onTabChange={setActiveTab}
          hasTools={!!(systemMessage.tools && systemMessage.tools.length > 0)}
        />

        <div className="max-h-[51vh] overflow-auto rounded-md custom-scrollbar-always">
          <TabContent
            activeTab={activeTab}
            systemMessage={systemMessage}
            expandedTools={expandedTools}
            onToggleTool={toggleTool}
          />
        </div>
      </div>
    </ModalBody>
  );
}
