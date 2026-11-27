/**
 * The action bridge — the only channel through which a manifest can make this
 * host *do* something.
 *
 * A manifest is data from another repository, so it never describes a request.
 * It chooses between the two outcomes this host offers, and it chooses by
 * declaring a `mode`: a direct entry produces a create request the host derives,
 * an assisted entry hands setup to a conversation.
 *
 * A direct entry that ships a bundle takes one more step before that create
 * request - packing and uploading the archive - but it still names no host, no
 * path and no method: the endpoints come from the interface manifest and the
 * files from the published package.
 */

import { useCallback, useRef } from "react";
import { initializeAutomationFormSession } from "#/api/automation-form-session";
import type { AutomationSetupDraft } from "#/api/automation-setup-types";
import AutomationService from "#/api/automation-service/automation-service.api";
import { markAutomationSetupHandoff } from "#/api/automation-setup-handoff-store";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { buildAutomationSetupContext } from "#/utils/automation-setup-context";
import {
  buildAssistedMessage,
  buildCreatePayload,
  isBundleEntry,
  isUploadAction,
} from "./automation-setup";
import { packBundle } from "./manifest-bundle";
import type {
  SetupEntry,
  SetupFormValue,
  SetupFormValues,
  SetupRequestBody,
} from "./types";

export interface SetupActionResult {
  /** The created resource, or the conversation that will finish setup. */
  response: Record<string, unknown>;
}

function isFileValue(value: SetupFormValue | undefined): value is File {
  return typeof File !== "undefined" && value instanceof File;
}

function uploadCacheKey(
  entry: SetupEntry,
  values: SetupFormValues,
  selectedAction?: string | null,
): string {
  const upload = values.tarball;
  const uploadIdentity = isFileValue(upload)
    ? {
        name: upload.name,
        size: upload.size,
        lastModified: upload.lastModified,
      }
    : upload;
  return `${entry.id}
${selectedAction ?? ""}
${JSON.stringify({ ...values, tarball: uploadIdentity })}`;
}

async function uploadedTarballBytes(
  values: SetupFormValues,
): Promise<Uint8Array> {
  const upload = values.tarball;
  if (!isFileValue(upload)) {
    throw new Error("Choose a tarball before creating this automation.");
  }
  return new Uint8Array(await upload.arrayBuffer());
}

export function useSetupAction() {
  const createConversation = useCreateConversation();

  // The last archive uploaded, and what the service called it. An upload that
  // is followed by a create the service rejects cannot be taken back - the
  // interface declares no endpoint for that - so confirming again after
  // correcting a field sends the archive that is already there rather than
  // leaving another copy of it behind. The archive is a pure function of the
  // entry and the answers, which is what makes the key sound.
  const uploadedRef = useRef<{ key: string; path: string } | null>(null);

  // An assisted template opens as an automation setup conversation: the seed
  // is sent as its first message and the draft editor opens beside it.
  const startConversation = useCallback(
    async (message: string): Promise<SetupActionResult> => {
      const draft: AutomationSetupDraft = { kind: "prompt", prompt: "" };
      const conversation = await createConversation.mutateAsync({
        query: message,
        automationSetup: true,
        automationSetupContext: buildAutomationSetupContext(draft),
        entryPoint: "automation_template",
      });
      initializeAutomationFormSession(conversation.conversation_id, draft);
      markAutomationSetupHandoff(conversation.conversation_id);
      return { response: { ...conversation } };
    },
    [createConversation],
  );

  return useCallback(
    async (
      entry: SetupEntry,
      values: SetupFormValues,
      /** Present for a direct entry, and absent for an assisted one. */
      payload: SetupRequestBody | null,
      selectedTrigger?: string | null,
      selectedAction?: string | null,
    ): Promise<SetupActionResult> => {
      if (!payload) {
        return startConversation(buildAssistedMessage(entry, values));
      }

      // A bundle entry ships a script rather than a prompt, so what it creates
      // from is an archive: pack it with the rendered config, upload it, and
      // create against the path that came back. The payload built for the form
      // carries a stand-in path, which is replaced here with the real one.
      if (isBundleEntry(entry) || isUploadAction(entry, selectedAction)) {
        const key = uploadCacheKey(entry, values, selectedAction);
        let tarballPath = uploadedRef.current?.path ?? null;
        if (uploadedRef.current?.key !== key || tarballPath === null) {
          const archive = isBundleEntry(entry)
            ? await packBundle(entry, values)
            : await uploadedTarballBytes(values);
          tarballPath = await AutomationService.uploadAutomationTarball(
            entry.id,
            archive,
          );
          uploadedRef.current = { key, path: tarballPath };
        }
        const body = buildCreatePayload(
          entry,
          values,
          tarballPath,
          selectedTrigger,
          selectedAction,
        );
        if (!body) throw new Error(`'${entry.id}' produced no create request.`);
        return {
          response: await AutomationService.createAutomationDraft(
            body,
            entry,
            selectedAction,
          ),
        };
      }

      const response = await AutomationService.createAutomationDraft(
        payload,
        entry,
        selectedAction,
      );
      return { response };
    },
    [startConversation],
  );
}
