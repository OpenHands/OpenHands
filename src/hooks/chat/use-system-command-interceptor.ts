import { useCallback } from "react";
import { useTranslation } from "react-i18next";
import { getLastRenderableEventId } from "#/hooks/chat/model-command-event-anchor";
import { useConversationSkills } from "#/hooks/query/use-conversation-skills";
import { useConversationHooks } from "#/hooks/query/use-conversation-hooks";
import { useSettings } from "#/hooks/query/use-settings";
import { useSlashCommandOutputStore } from "#/stores/slash-command-output-store";
import {
  BUILT_IN_COMMANDS,
  HELP_COMMAND,
  SKILLS_COMMAND,
} from "#/utils/constants";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";
import { flattenMcpConfig } from "#/utils/mcp-installed-servers";

/**
 * Intercepts browser-local utility commands. These commands never reach the
 * agent as user messages; help and skills produce anchored inline chat cards
 * listing available commands and loaded extensions.
 */
export function useSystemCommandInterceptor(
  conversationId: string | null | undefined,
  onSubmit: (message: string) => void,
) {
  const { t } = useTranslation("openhands");
  const { data: skills, refetch: refetchSkills } = useConversationSkills();
  const { data: hooks, refetch: refetchHooks } =
    useConversationHooks(conversationId);
  const { data: settings, refetch: refetchSettings } = useSettings();
  const store = useSlashCommandOutputStore();
  const showHelp = store.showHelp;
  const showSkills = store.showSkills;

  return useCallback(
    (message: string) => {
      const command = message.trim();
      const isSystemCommand = [HELP_COMMAND, SKILLS_COMMAND].includes(command);
      if (!isSystemCommand) {
        onSubmit(message);
        return;
      }

      if (!conversationId) {
        displayErrorToast(
          t(I18nKey.SLASH_COMMAND$ACTIVE_CONVERSATION_REQUIRED),
        );
        return;
      }

      const anchorEventId = getLastRenderableEventId();

      // @spec SC-002 — Inline help
      if (command === HELP_COMMAND) {
        showHelp(conversationId, anchorEventId, BUILT_IN_COMMANDS);
        return;
      }

      // @spec SC-003 — Loaded extensions
      Promise.all([refetchSkills(), refetchHooks(), refetchSettings()])
        .then(([skillsResult, hooksResult, settingsResult]) => {
          const currentSkills = skillsResult.isError
            ? (skills ?? [])
            : (skillsResult.data ?? skills ?? []);
          const currentHooks = hooksResult.isError
            ? (hooks ?? [])
            : (hooksResult.data ?? hooks ?? []);
          const currentSettings = settingsResult.isError
            ? settings
            : (settingsResult.data ?? settings);
          const mcpConfig = currentSettings?.mcp_config;
          const mcpServers = mcpConfig
            ? flattenMcpConfig(mcpConfig).filter(
                (server) => server.enabled !== false,
              )
            : [];

          showSkills(conversationId, anchorEventId, {
            skills: currentSkills,
            hooks: currentHooks,
            mcpServers,
          });
        })
        .catch(() => displayErrorToast(t(I18nKey.ERROR$GENERIC)));
    },
    [
      conversationId,
      hooks,
      onSubmit,
      refetchHooks,
      refetchSettings,
      refetchSkills,
      settings?.mcp_config,
      showHelp,
      showSkills,
      skills,
      t,
    ],
  );
}
