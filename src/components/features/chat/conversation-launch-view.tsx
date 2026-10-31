import { CustomChatInput } from "./custom-chat-input";
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
    <div
      className="mx-auto flex h-full w-full max-w-200 flex-col px-8 py-4"
      data-testid="conversation-launch-view"
    >
      {launch ? (
        <>
          <div
            data-testid="startup-user-message"
            className="mt-6 w-fit max-w-full self-end whitespace-pre-wrap break-words rounded-xl bg-tertiary px-4 py-2.5 text-sm leading-6"
          >
            {launch.message}
            {launch.files.length > 0 && (
              <ul>
                {launch.files.map((file, index) => (
                  <li key={`${index}-${file.name}`}>{file.name}</li>
                ))}
              </ul>
            )}
          </div>
          <ConversationStartupProgress
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
                  if (!composer.imagesMarkedUploadAsFile.includes(name))
                    composer.toggleImageUploadAsFile(name);
                }
              }
              editing.current = true;
              navigate("/conversations");
            }}
          />
        </>
      ) : (
        <div role="status" className="flex flex-col items-start gap-4">
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
      {launch && !launch.error && (
        <div className="mt-auto pt-4">
          <CustomChatInput
            disabled
            persistDraft={false}
            hasStartedConversation
            onSubmit={() => {}}
          />
        </div>
      )}
    </div>
  );
}
