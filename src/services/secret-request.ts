import type { CustomSecretInput } from "#/api/agent-server-adapter";
import type { CloudSecretAttachResult } from "#/api/cloud/conversation-service.api";
import {
  REQUEST_SECRET_TOOL_NAME,
  SECRET_REQUEST_RESULT_PREFIX,
} from "#/constants/secret-request";
import { attachConversationSecret } from "#/hooks/mutation/conversation-mutation-utils";
import {
  type SecretRequest,
  useSecretRequestStore,
} from "#/stores/secret-request-store";
import type { RequestSecretAction } from "#/types/agent-server/core";
import { claimClientToolCall } from "#/utils/claim-client-tool-call";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { postClientToolResult } from "./client-tool-result";

const LEDGER_STORAGE_KEY_PREFIX = "openhands-secret-requests:";

type SecretRequestResult =
  | { status: "saved"; name: string }
  | { status: "not_saved"; name: string; guidance: string }
  | { status: "error"; name: string; error: string; guidance: string };

/** Queue a secure form; only an explicit user save can report success. */
export async function handleRequestSecretAction(
  action: RequestSecretAction,
  conversationId: string,
  toolCallId: string,
): Promise<void> {
  // A replayed call would ask again and report the outcome twice.
  if (
    !claimClientToolCall(
      `${LEDGER_STORAGE_KEY_PREFIX}${conversationId}`,
      toolCallId,
    )
  ) {
    return;
  }

  const request: SecretRequest = {
    conversationId,
    toolCallId,
    name: action.name,
    reason: action.reason ?? null,
  };
  useSecretRequestStore.getState().enqueue(request);
}

async function resolveResult(
  request: SecretRequest,
  savedSecret: CustomSecretInput | null,
): Promise<SecretRequestResult> {
  if (!savedSecret) {
    return {
      status: "not_saved",
      name: request.name,
      guidance:
        "The user dismissed the form without saving the secret. Ask them how they want to proceed; never ask them to paste it into the chat.",
    };
  }

  const { name } = savedSecret;
  const notAttachedGuidance =
    "The secret was saved to the user's settings but could not be attached to this conversation. It will be available in new conversations; tell the user.";
  let attach: CloudSecretAttachResult;
  try {
    attach = await attachConversationSecret(
      request.conversationId,
      savedSecret,
    );
  } catch (error) {
    return {
      status: "error",
      name,
      error: error instanceof Error ? error.message : String(error),
      guidance: notAttachedGuidance,
    };
  }

  switch (attach.status) {
    case "attached":
      return { status: "saved", name };
    case "deferred":
      return {
        status: "not_saved",
        name,
        guidance:
          "The user saved the secret to their settings, but this conversation's sandbox was not running, so it was not attached to this conversation and is not set in your environment. It will be available in new conversations. Tell the user; never ask them to paste it into the chat.",
      };
    case "not_found":
      return {
        status: "error",
        name,
        error: `The saved secret ${name} could not be found.`,
        guidance:
          "The secret could not be found in the user's saved secrets, so it was not attached to this conversation. Tell the user to check it in Settings > Secrets; never ask them to paste it into the chat.",
      };
    default:
      return {
        status: "error",
        name,
        error:
          attach.message ||
          `The secret ${name} could not be attached to this conversation.`,
        guidance: notAttachedGuidance,
      };
  }
}

/**
 * Hand the outcome of a secret request back to the agent. It never carries the
 * value. Each request is reported once, even if the form settles after cancel.
 *
 * Never rejects: the user has already answered, so a failure to report is
 * logged rather than surfaced.
 */
export async function completeSecretRequest(
  request: SecretRequest,
  savedSecret: CustomSecretInput | null,
): Promise<void> {
  const store = useSecretRequestStore.getState();
  if (!store.requests.some((r) => r.toolCallId === request.toolCallId)) return;
  store.dismiss(request.toolCallId);

  const result = await resolveResult(request, savedSecret);
  // Tell the user even when the agent cannot be told (an active `/goal` loop).
  if (result.status === "error") displayErrorToast(result.error);

  await postClientToolResult(
    request.conversationId,
    SECRET_REQUEST_RESULT_PREFIX,
    result,
  ).catch((error) => {
    console.warn(
      `[${REQUEST_SECRET_TOOL_NAME}] Failed to report the request result:`,
      error,
    );
  });
}
