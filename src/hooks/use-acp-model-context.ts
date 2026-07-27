import { useTranslation } from "react-i18next";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useSettings } from "#/hooks/query/use-settings";
import { useActiveAgentKind } from "#/hooks/use-active-agent-profile";
import { I18nKey } from "#/i18n/declaration";

const AGENT_SETTINGS_PATH = "/settings/agents";
const LLM_SETTINGS_PATH = "/settings/llm";

export interface AcpModelContext {
  /** The active conversation runs an ACP agent. */
  isActiveAcpConversation: boolean;
  /**
   * No active conversation (home page) but the saved agent settings already
   * select an ACP agent — the next conversation created here inherits it, so
   * the model UI should reflect that now.
   */
  isHomeAcp: boolean;
  /** Either of the above: the model affordance should defer to the ACP picker. */
  isAcpContext: boolean;
  /** Where the model/settings link should navigate. */
  destinationPath: typeof AGENT_SETTINGS_PATH | typeof LLM_SETTINGS_PATH;
  /** Translated label for that link. */
  destinationLabel: string;
}

/**
 * Shared source of truth for "is this an ACP model context, and where does
 * the model affordance link?". The chat model affordance consumes this through
 * ``useChatInputModelState`` so inline and overflow surfaces can't drift on the
 * home-page-ACP rule or on the destination path/label.
 */
export function useAcpModelContext(): AcpModelContext {
  const { t } = useTranslation("openhands");
  const { backend } = useActiveBackend();
  const { data: conversation } = useActiveConversation();
  const { data: settings } = useSettings();
  const activeAgentKind = useActiveAgentKind();

  const isActiveAcpConversation = conversation?.agent_kind === "acp";
  // On the home page the "current agent" is the active AgentProfile, not the
  // global agent settings (activate never writes them). Fall back to settings
  // only while the profile list is loading.
  const isHomeAcp =
    !conversation &&
    (activeAgentKind ?? settings?.agent_settings?.agent_kind) === "acp";
  const isAcpContext = isActiveAcpConversation || isHomeAcp;

  const destinationPath = isAcpContext
    ? AGENT_SETTINGS_PATH
    : LLM_SETTINGS_PATH;
  const destinationLabel = t(
    isAcpContext
      ? I18nKey.SETTINGS$NAV_AGENT
      : backend.kind === "cloud"
        ? I18nKey.SETTINGS$LLM_SETTINGS
        : I18nKey.SETTINGS$LLM_PROFILES,
  );

  return {
    isActiveAcpConversation,
    isHomeAcp,
    isAcpContext,
    destinationPath,
    destinationLabel,
  };
}
