import { useEffect } from "react";
import { BrowserSnapshot } from "./browser-snapshot";
import { EmptyBrowserMessage } from "./empty-browser-message";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useBrowserStore } from "#/stores/browser-store";

export function BrowserPanel() {
  const { url, screenshotSrc, reset } = useBrowserStore();
  const { conversationId } = useConversationId();

  useEffect(() => {
    reset();
  }, [conversationId, reset]);

  const imgSrc = screenshotSrc?.startsWith("data:image/png;base64,")
    ? screenshotSrc
    : `data:image/png;base64,${screenshotSrc ?? ""}`;

  return (
    <div className="h-full w-full flex flex-col text-neutral-400">
      <div className="w-full p-2 truncate border-b border-neutral-600">
        {url}
      </div>
      <div className="overflow-y-auto grow scrollbar-hide rounded-xl">
        {screenshotSrc ? (
          <BrowserSnapshot src={imgSrc} />
        ) : (
          <EmptyBrowserMessage />
        )}
      </div>
    </div>
  );
}
