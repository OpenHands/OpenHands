import { useMemo } from "react";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useSettings } from "#/hooks/query/use-settings";
import { extractModelAndProvider } from "#/utils/extract-model-and-provider";
import { resolveUploadLimits } from "#/utils/file-validation";

/**
 * The upload limits that apply to the attachments the user is adding now.
 *
 * The limit belongs to the provider that will receive the attachment, so it is
 * resolved from the model the active conversation runs (falling back to the
 * configured default model on the home screen, before a conversation exists).
 */
export function useCurrentUploadLimits() {
  const { data: conversation } = useActiveConversation();
  const { data: settings } = useSettings();

  return useMemo(() => {
    const model = conversation?.llm_model ?? settings?.llm_model ?? null;
    const { provider } = model
      ? extractModelAndProvider(model)
      : { provider: null };

    return resolveUploadLimits(provider || null);
  }, [conversation?.llm_model, settings?.llm_model]);
}
