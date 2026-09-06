import { useEffect, useRef } from "react";
import type { ProfileInfo } from "#/api/profiles-service/profiles-service.api";
import {
  LLM_AUTH_TYPE_SUBSCRIPTION,
  OPENAI_SUBSCRIPTION_VENDOR,
} from "#/constants/llm-subscription";
import { useSaveLlmProfile } from "#/hooks/mutation/use-save-llm-profile";
import { useSubscriptionModelEnablement } from "#/hooks/use-subscription-model-enablement";
import {
  chatgptAutoProfileName,
  type SubscriptionModelOffer,
} from "#/utils/subscription-model-catalog";

/**
 * Creates ChatGPT LLM profiles for enabled catalog models so they appear in
 * the chat picker without a manual "Add profile" click.
 */
export function useSyncChatgptSubscriptionProfiles(
  offers: SubscriptionModelOffer[],
  profiles: ProfileInfo[],
) {
  const { isOfferEnabled } = useSubscriptionModelEnablement();
  const saveProfile = useSaveLlmProfile();
  const inFlight = useRef(false);

  useEffect(() => {
    if (inFlight.current) return;
    const pending = offers.filter((offer) => {
      if (offer.source !== "chatgpt" || !isOfferEnabled(offer)) return false;
      const name = chatgptAutoProfileName(offer.id);
      return !profiles.some(
        (profile) => profile.name === name || profile.model === offer.id,
      );
    });
    if (pending.length === 0) return;

    inFlight.current = true;
    void (async () => {
      try {
        for (const offer of pending) {
          await saveProfile.mutateAsync({
            name: chatgptAutoProfileName(offer.id),
            request: {
              llm: {
                auth_type: LLM_AUTH_TYPE_SUBSCRIPTION,
                subscription_vendor: OPENAI_SUBSCRIPTION_VENDOR,
                model: offer.id,
              },
            },
          });
        }
      } catch {
        // Individual saves toast at the call site; leave the rest for next load.
      } finally {
        inFlight.current = false;
      }
    })();
  }, [isOfferEnabled, offers, profiles, saveProfile]);
}
