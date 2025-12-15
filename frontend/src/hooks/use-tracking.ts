import { usePostHog } from "posthog-js/react";
import { useConfig } from "./query/use-config";
import { useSettings } from "./query/use-settings";
import { Provider } from "#/types/settings";

/**
 * Hook that provides tracking functions with automatic data collection
 * from available hooks (config, settings, etc.)
 */
export const useTracking = () => {
  const posthog = usePostHog();
  const { data: config } = useConfig();
  const { data: settings } = useSettings();

  // Common properties included in all tracking events
  const commonProperties = {
    app_surface: config?.APP_MODE || "unknown",
    plan_tier: null,
    current_url: window.location.href,
    user_email: settings?.email || settings?.git_user_email || null,
  };

  const trackLoginButtonClick = ({ provider }: { provider: Provider }) => {
    posthog.capture("login_button_clicked", {
      provider,
      ...commonProperties,
    });
  };

  const trackConversationCreated = ({
    hasRepository,
  }: {
    hasRepository: boolean;
  }) => {
    posthog.capture("conversation_created", {
      has_repository: hasRepository,
      ...commonProperties,
    });
  };

  const trackPushButtonClick = () => {
    posthog.capture("push_button_clicked", {
      ...commonProperties,
    });
  };

  const trackPullButtonClick = () => {
    posthog.capture("pull_button_clicked", {
      ...commonProperties,
    });
  };

  const trackCreatePrButtonClick = () => {
    posthog.capture("create_pr_button_clicked", {
      ...commonProperties,
    });
  };

  const trackGitProviderConnected = ({
    providers,
  }: {
    providers: string[];
  }) => {
    posthog.capture("git_provider_connected", {
      providers,
      ...commonProperties,
    });
  };

  const trackUserSignupCompleted = () => {
    posthog.capture("user_signup_completed", {
      signup_timestamp: new Date().toISOString(),
      ...commonProperties,
    });
  };

  const trackCreditsPurchased = ({
    amountUsd,
    stripeSessionId,
  }: {
    amountUsd: number;
    stripeSessionId: string;
  }) => {
    posthog.capture("credits_purchased", {
      amount_usd: amountUsd,
      stripe_session_id: stripeSessionId,
      ...commonProperties,
    });
  };

  const trackCreditLimitReached = ({
    conversationId,
  }: {
    conversationId: string;
  }) => {
    posthog.capture("credit_limit_reached", {
      conversation_id: conversationId,
      ...commonProperties,
    });
  };

  return {
    trackLoginButtonClick,
    trackConversationCreated,
    trackPushButtonClick,
    trackPullButtonClick,
    trackCreatePrButtonClick,
    trackGitProviderConnected,
    trackUserSignupCompleted,
    trackCreditsPurchased,
    trackCreditLimitReached,
  };
};
