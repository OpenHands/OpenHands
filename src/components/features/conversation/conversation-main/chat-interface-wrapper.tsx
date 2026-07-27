import { ChatInterface } from "../../chat/chat-interface";

interface ChatInterfaceWrapperProps {
  isRightPanelShown: boolean;
}

export function ChatInterfaceWrapper({
  isRightPanelShown: _isRightPanelShown,
}: ChatInterfaceWrapperProps) {
  return (
    <div className="flex justify-center w-full h-full">
      <div className="w-full min-w-0 max-w-[800px] h-full flex flex-col min-h-0 transition-all duration-300 ease-in-out">
        <ChatInterface />
      </div>
    </div>
  );
}
