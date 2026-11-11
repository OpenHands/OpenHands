import { useCallback } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { getApiErrorMessage } from "#/utils/api-error-message";
import { useConversationLaunchStore } from "#/stores/conversation-launch-store";
import { buildAutomationSetupContext } from "#/utils/automation-setup-context";
import {
  PENDING_AUTOMATION_SETUP_ID,
  clearAutomationFormSession,
  getAutomationFormSession,
  initializeAutomationFormSession,
} from "#/api/automation-form-session";
import { markAutomationSetupHandoff } from "#/api/automation-setup-handoff-store";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useNavigation } from "#/context/navigation-context";
import { useCreateAutomationSetupConversation } from "#/hooks/use-create-automation-setup-conversation";
import { useTracking } from "#/hooks/use-tracking";
import {
  buildAutomationDraftTags,
  buildAutomationEditTags,
} from "#/utils/automation-draft-tags";

/**
 * Open a new automation on the setup form.
 *
 * The automations page used to create an agent conversation immediately.
 * The form now opens with the conversation drawer hidden, and a conversation
 * starts only after the user sends a prompt.
 */
export function useStartAutomationSetup() {
  const { t } = useTranslation("openhands");
  const active = useActiveBackend();
  const { navigate } = useNavigation();
  const { startAutomationSetupConversation, isPending } =
    useCreateAutomationSetupConversation();
  const { trackAutomationCreatedButton } = useTracking();

  const startSetup = useCallback(() => {
    trackAutomationCreatedButton({ backendKind: active.backend.kind });
    clearAutomationFormSession(PENDING_AUTOMATION_SETUP_ID);
    navigate?.("/automations/setup");
  }, [active.backend.kind, navigate, trackAutomationCreatedButton]);

  const startConversationFromPrompt = useCallback(
    (prompt: string) => {
      if (!prompt.trim()) return;
      const draft = getAutomationFormSession(PENDING_AUTOMATION_SETUP_ID) ?? {
        prompt: "",
        kind: "prompt" as const,
      };
      const editingAutomationId = draft.editingAutomationId?.trim() || null;
      let entryPoint:
        "automations_add" | "automation_edit" | "automation_draft_resume" =
        "automations_add";
      let automationSetupTags: Record<string, string> | undefined;
      if (editingAutomationId) {
        entryPoint = "automation_edit";
        automationSetupTags = buildAutomationEditTags(
          null,
          editingAutomationId,
        );
      } else if (draft.serverDraftId) {
        entryPoint = "automation_draft_resume";
        automationSetupTags = buildAutomationDraftTags(
          null,
          draft.serverDraftId,
          draft.materializedAutomationId,
        );
      }
      let launchId: string;
      startAutomationSetupConversation({
        query: prompt,
        automationSetupContext: buildAutomationSetupContext(draft),
        entryPoint,
        automationSetupTags,
        onStart: () => {
          launchId = useConversationLaunchStore.getState().begin({
            backendId: active.backend.id,
            orgId: active.orgId ?? null,
            message: prompt.trim(),
            files: [],
            automationSetupDraft: draft,
          });
          clearAutomationFormSession(PENDING_AUTOMATION_SETUP_ID);
          navigate?.(`/conversations/${launchId}`);
        },
        onSuccess: (conversation) => {
          const conversationId = conversation.conversation_id;
          initializeAutomationFormSession(conversationId, draft);
          markAutomationSetupHandoff(conversationId);
          useConversationLaunchStore.getState().update(launchId, {
            targetId: conversationId,
          });
        },
        onError: (error) => {
          useConversationLaunchStore.getState().update(launchId, {
            error: getApiErrorMessage(error, t(I18nKey.ERROR$GENERIC)),
          });
        },
      });
    },
    [
      active.backend.id,
      active.orgId,
      navigate,
      startAutomationSetupConversation,
      t,
    ],
  );

  return {
    startSetup,
    startConversationFromPrompt,
    isPending,
  };
}
