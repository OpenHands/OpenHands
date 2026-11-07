import { InteractiveChatBox } from "./interactive-chat-box";
import { ChatMessage } from "./chat-message";
import { ConversationMain } from "../conversation/conversation-main/conversation-main";
import { ConversationOverviewDrawerProvider } from "../conversation/conversation-overview-drawer-context";
import { ChatProgressContext } from "#/context/chat-progress-context";
import {
  CHAT_BODY_CLASS_NAME,
  CHAT_SCROLL_CLASS_NAME,
  CHAT_COMPOSER_CLASS_NAME,
} from "./chat-layout";
import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import { useConversationLaunchStore } from "#/stores/conversation-launch-store";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useNavigation } from "#/context/navigation-context";
import { useConversationStore } from "#/stores/conversation-store";
import { I18nKey } from "#/i18n/declaration";
import { ConversationStartupProgress } from "./conversation-startup-progress";

export function ConversationLaunchView({ launchId }: { launchId: string }) {
  const { t } = useTranslation("openhands");
  const { backend, orgId } = useActiveBackend();
  const { navigate } = useNavigation();
  const entry = useConversationLaunchStore((state) => state.launches[launchId]);
  const launch =
    entry?.backendId === backend.id && entry.orgId === orgId ? entry : null;
  const editing = useRef(false);
  useEffect(
    () => () => {
      if (editing.current)
        useConversationLaunchStore.getState().remove(launchId);
    },
    [launchId],
  );
  useEffect(() => {
    if (launch?.targetId) {
      navigate(`/conversations/${launch.targetId}`, { replace: true });
    }
  }, [launch?.targetId, navigate]);

  return (
    <ConversationOverviewDrawerProvider>
      <div
        className="flex h-full min-h-0 flex-col"
        data-testid="conversation-launch-view"
      >
        <ConversationMain>
          <ChatProgressContext.Provider value>
            <div className={CHAT_BODY_CLASS_NAME}>
              <div className={CHAT_SCROLL_CLASS_NAME}>
                {launch ? (
                  <>
                    <ChatMessage type="user" message={launch.message}>
                      {launch.files.length > 0 && (
                        <ul>
                          {launch.files.map((file, index) => (
                            <li key={`${index}-${file.name}`}>{file.name}</li>
                          ))}
                        </ul>
                      )}
                    </ChatMessage>
                    <ConversationStartupProgress
                      key={launch.id}
                      error={launch.error}
                      onEdit={() => {
                        const composer = useConversationStore.getState();
                        composer.setMessageToSend(launch.message, "home");
                        if (launch.attachments) {
                          composer.addImages(
                            launch.attachments.images.filter(
                              (file) => !composer.images.includes(file),
                            ),
                          );
                          composer.addFiles(
                            launch.attachments.files.filter(
                              (file) => !composer.files.includes(file),
                            ),
                          );
                          for (const name of launch.attachments
                            .imagesMarkedUploadAsFile) {
                            if (
                              !composer.imagesMarkedUploadAsFile.includes(name)
                            )
                              composer.toggleImageUploadAsFile(name);
                          }
                        }
                        editing.current = true;
                        navigate("/conversations");
                      }}
                    />
                  </>
                ) : (
                  <div
                    role="status"
                    className="flex flex-col items-start gap-4"
                  >
                    <p>{t(I18nKey.CONVERSATION$START_NOT_IN_TAB)}</p>
                    <button
                      type="button"
                      className="button-base rounded-md border border-border px-3 py-1.5"
                      onClick={() => navigate("/conversations")}
                    >
                      {t(I18nKey.COMMON$BACK)}
                    </button>
                  </div>
                )}
              </div>
              {launch && (
                <div className={CHAT_COMPOSER_CLASS_NAME}>
                  <InteractiveChatBox
                    disabled
                    persistDraft={false}
                    hasStartedConversation
                    onSubmit={() => {}}
                  />
                </div>
              )}
            </div>
          </ChatProgressContext.Provider>
        </ConversationMain>
      </div>
    </ConversationOverviewDrawerProvider>
  );
}
