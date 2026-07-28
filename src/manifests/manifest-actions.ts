/**
 * The action bridge — the only channel through which a manifest can make this
 * host *do* something.
 *
 * A manifest is data from another repository, so it chooses among pre-approved
 * capabilities rather than describing a request of its own. The union of
 * actions is closed at the type level and admission rejects anything outside
 * it, which is what keeps the manifest data rather than code.
 */

import { useCallback } from "react";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { useConversationStore } from "#/stores/conversation-store";
import {
  setConversationState,
  setPendingTaskDraft,
} from "#/utils/conversation-local-storage";
import { buildRequestBody, interpolateText } from "./manifest-template";
import type {
  ExtensionManifest,
  ManifestFormValues,
  ManifestRequestBody,
} from "./types";

export interface ManifestActionResult {
  /** Values the manifest may reference through the `response` namespace. */
  response: Record<string, unknown>;
}

/**
 * Map form values into the request body the manifest declares.
 *
 * Returns null for a manifest that hands setup to a conversation instead of
 * sending a body. The result is also what preflight validates, so what is
 * checked is exactly what would be sent.
 */
export function buildManifestPayload(
  manifest: ExtensionManifest,
  formValues: ManifestFormValues,
): ManifestRequestBody | null {
  if (manifest.submit.action !== "automation.create") return null;
  return buildRequestBody(manifest.submit.payload, {
    form: formValues,
    manifest,
  });
}

export function useManifestAction() {
  const createConversation = useCreateConversation();
  const setMessageToSend = useConversationStore(
    (state) => state.setMessageToSend,
  );

  const startConversation = useCallback(
    async (message: string): Promise<ManifestActionResult> => {
      const conversation = await createConversation.mutateAsync({});

      // Seed the message the same way the rest of the app does, so the
      // conversation opens with it queued whichever launch path applies.
      if (
        conversation.conversation_id.startsWith("task-") &&
        conversation.task_id
      ) {
        setPendingTaskDraft(conversation.task_id, message);
      } else {
        setConversationState(conversation.conversation_id, {
          draftMessage: message,
        });
      }
      window.setTimeout(() => setMessageToSend(message), 0);

      return { response: { ...conversation } };
    },
    [createConversation, setMessageToSend],
  );

  return useCallback(
    async (
      manifest: ExtensionManifest,
      formValues: ManifestFormValues,
      payload: ManifestRequestBody | null,
    ): Promise<ManifestActionResult> => {
      const { submit } = manifest;

      if (submit.action === "conversation.start") {
        return startConversation(
          interpolateText(submit.message, { form: formValues, manifest }),
        );
      }

      const response = await AutomationService.createFromManifest(
        submit.endpoint.path,
        payload ??
          buildRequestBody(submit.payload, { form: formValues, manifest }),
      );
      return { response };
    },
    [startConversation],
  );
}
