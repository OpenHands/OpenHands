import { useCallback } from "react";
import {
  LLM_AUTH_TYPE_SUBSCRIPTION,
  OPENAI_SUBSCRIPTION_VENDOR,
} from "#/constants/llm-subscription";
import { useDeleteLlmProfile } from "#/hooks/mutation/use-delete-llm-profile";
import { useSaveLlmProfile } from "#/hooks/mutation/use-save-llm-profile";
import {
  isSubscriptionModelEnabled,
  useSubscriptionModelPreferencesStore,
} from "#/stores/subscription-model-preferences-store";
import {
  chatgptAutoProfileName,
  subscriptionModelToggleKey,
  type SubscriptionModelOffer,
} from "#/utils/subscription-model-catalog";

export function useSubscriptionModelEnablement() {
  const disabledKeys = useSubscriptionModelPreferencesStore(
    (state) => state.disabledKeys,
  );
  const setEnabled = useSubscriptionModelPreferencesStore(
    (state) => state.setEnabled,
  );
  const saveProfile = useSaveLlmProfile();
  const deleteProfile = useDeleteLlmProfile();

  const isOfferEnabled = useCallback(
    (offer: SubscriptionModelOffer) =>
      isSubscriptionModelEnabled(
        disabledKeys,
        subscriptionModelToggleKey(offer.source, offer.id),
      ),
    [disabledKeys],
  );

  const setOfferEnabled = useCallback(
    async (offer: SubscriptionModelOffer, enabled: boolean) => {
      setEnabled(subscriptionModelToggleKey(offer.source, offer.id), enabled);
      if (offer.source !== "chatgpt") return;
      const name = chatgptAutoProfileName(offer.id);
      if (enabled) {
        await saveProfile.mutateAsync({
          name,
          request: {
            llm: {
              auth_type: LLM_AUTH_TYPE_SUBSCRIPTION,
              subscription_vendor: OPENAI_SUBSCRIPTION_VENDOR,
              model: offer.id,
            },
          },
        });
        return;
      }
      try {
        await deleteProfile.mutateAsync(name);
      } catch {
        // The auto profile may not exist yet (never synced).
      }
    },
    [deleteProfile, saveProfile, setEnabled],
  );

  return { isOfferEnabled, setOfferEnabled };
}
