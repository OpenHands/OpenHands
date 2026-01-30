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

  const trackDownloadViaVSCodeButtonClick = () => {
    posthog.capture("download_via_vscode_button_clicked", {
      ...commonProperties,
    });
  };

  const trackDownloadTrajectoryButtonClick = () => {
    posthog.capture("download_trajectory_button_clicked", {
      ...commonProperties,
    });
  };

  const trackMcpConfigUpdated = ({
    hasMcpConfig,
    sseServersCount,
    stdioServersCount,
  }: {
    hasMcpConfig: boolean;
    sseServersCount: number;
    stdioServersCount: number;
  }) => {
    posthog.capture("mcp_config_updated", {
      has_mcp_config: hasMcpConfig,
      sse_servers_count: sseServersCount,
      stdio_servers_count: stdioServersCount,
      ...commonProperties,
    });
  };

  const trackSettingsSaved = ({
    llmModel,
    llmApiKeySet,
    searchApiKeySet,
    remoteRuntimeResourceFactor,
  }: {
    llmModel: string | undefined;
    llmApiKeySet: string;
    searchApiKeySet: string;
    remoteRuntimeResourceFactor: number | null | undefined;
  }) => {
    posthog.capture("settings_saved", {
      LLM_MODEL: llmModel,
      LLM_API_KEY_SET: llmApiKeySet,
      SEARCH_API_KEY_SET: searchApiKeySet,
      REMOTE_RUNTIME_RESOURCE_FACTOR: remoteRuntimeResourceFactor,
      ...commonProperties,
    });
  };

  const trackInitialQuerySubmitted = ({
    entryPoint,
    queryCharacterLength,
    replayJsonSize,
  }: {
    entryPoint: string;
    queryCharacterLength: number;
    replayJsonSize: number | undefined;
  }) => {
    posthog.capture("initial_query_submitted", {
      entry_point: entryPoint,
      query_character_length: queryCharacterLength,
      replay_json_size: replayJsonSize,
      ...commonProperties,
    });
  };

  const trackUserMessageSent = ({
    sessionMessageCount,
    currentMessageLength,
  }: {
    sessionMessageCount: number;
    currentMessageLength: number;
  }) => {
    posthog.capture("user_message_sent", {
      session_message_count: sessionMessageCount,
      current_message_length: currentMessageLength,
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
    trackDownloadViaVSCodeButtonClick,
    trackDownloadTrajectoryButtonClick,
    trackMcpConfigUpdated,
    trackSettingsSaved,
    trackInitialQuerySubmitted,
    trackUserMessageSent,
  };
};
