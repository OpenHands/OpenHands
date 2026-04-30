import { useState } from "react";
import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { SystemMessageHeader } from "./system-message-modal/system-message-header";
import { TabNavigation } from "./system-message-modal/tab-navigation";
import { TabContent } from "./system-message-modal/tab-content";
import { SystemMessageForModal } from "#/utils/system-message-adapter";
import { I18nKey } from "#/i18n/declaration";
import { getToolDisplayMetadata } from "./system-message-modal/tool-item";
import { BrandButton } from "../settings/brand-button";

interface SystemMessageModalProps {
  isOpen: boolean;
  onClose: () => void;
  systemMessage: SystemMessageForModal | null;
}

export function SystemMessageModal({
  isOpen,
  onClose,
  systemMessage,
}: SystemMessageModalProps) {
  const { t } = useTranslation();
  const [activeTab, setActiveTab] = useState<"system" | "tools">("system");
  const [toolSearchQuery, setToolSearchQuery] = useState("");
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

  const tools = systemMessage.tools ?? [];
  const normalizedSearch = toolSearchQuery.trim().toLowerCase();
  const filteredTools = tools
    .map((tool, index) => ({
      tool,
      index,
      metadata: getToolDisplayMetadata(tool),
    }))
    .filter(({ metadata }) => {
      if (!normalizedSearch) {
        return true;
      }

      return (
        metadata.name.toLowerCase().includes(normalizedSearch) ||
        metadata.description.toLowerCase().includes(normalizedSearch) ||
        metadata.kind?.toLowerCase().includes(normalizedSearch)
      );
    })
    .map(({ tool, index }) => ({ tool, index }));

  const allExpanded =
    filteredTools.length > 0 &&
    filteredTools.every(({ index }) => expandedTools[index]);

  const toggleAllTools = () => {
    if (!filteredTools.length) {
      return;
    }

    setExpandedTools((prev) => {
      const next = { ...prev };
      const shouldExpand = !allExpanded;

      filteredTools.forEach(({ index }) => {
        next[index] = shouldExpand;
      });

      return next;
    });
  };

  return (
    isOpen && (
      <ModalBackdrop onClose={onClose}>
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
              hasTools={(systemMessage.tools?.length ?? 0) > 0}
              toolCount={filteredTools.length}
            />

            {activeTab === "tools" && tools.length > 0 && (
              <div className="mb-3 flex items-center gap-2">
                <input
                  type="text"
                  value={toolSearchQuery}
                  onChange={(event) => setToolSearchQuery(event.target.value)}
                  placeholder={t(
                    I18nKey.SYSTEM_MESSAGE_MODAL$SEARCH_TOOLS_PLACEHOLDER,
                  )}
                  className="flex-1 px-3 py-2 bg-gray-900 border border-gray-700 rounded-md text-sm text-gray-100 placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-primary"
                />
                <BrandButton
                  type="button"
                  variant="secondary"
                  onClick={toggleAllTools}
                  isDisabled={filteredTools.length === 0}
                >
                  {allExpanded
                    ? t(I18nKey.SYSTEM_MESSAGE_MODAL$COLLAPSE_ALL)
                    : t(I18nKey.SYSTEM_MESSAGE_MODAL$EXPAND_ALL)}
                </BrandButton>
              </div>
            )}

            <div className="max-h-[51vh] overflow-auto rounded-md custom-scrollbar-always">
              <TabContent
                activeTab={activeTab}
                systemMessage={systemMessage}
                expandedTools={expandedTools}
                filteredTools={filteredTools}
                onToggleTool={toggleTool}
              />
            </div>
          </div>
        </ModalBody>
      </ModalBackdrop>
    )
  );
}
