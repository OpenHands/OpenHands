import { useCallback } from "react";
import { useTranslation } from "react-i18next";
import { getLastRenderableEventId } from "#/hooks/chat/model-command-event-anchor";
import { buildSlashCommandItems } from "#/hooks/chat/use-slash-command";
import { condenseConversation } from "#/hooks/mutation/conversation-mutation-utils";
import { useConversationHooks } from "#/hooks/query/use-conversation-hooks";
import { useConversationSkills } from "#/hooks/query/use-conversation-skills";
import { useSettings } from "#/hooks/query/use-settings";
import { useSlashCommandOutputStore } from "#/stores/slash-command-output-store";
import {
  CONDENSE_COMMAND,
  FEEDBACK_COMMAND,
  FEEDBACK_FORM_URL,
  HELP_COMMAND,
  SKILLS_COMMAND,
} from "#/utils/constants";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";
import { flattenMcpConfig } from "#/utils/mcp-installed-servers";

const CONDENSE_UNSUPPORTED_STATUS_CODES = new Set([404, 405, 501]);

function getHttpStatus(error: unknown): number | undefined {
  if (typeof error !== "object" || error === null) return undefined;

  const directStatus = (error as { status?: unknown }).status;
  if (typeof directStatus === "number") return directStatus;

  const response = (error as { response?: unknown }).response;
  if (typeof response !== "object" || response === null) return undefined;
  const responseStatus = (response as { status?: unknown }).status;
  return typeof responseStatus === "number" ? responseStatus : undefined;
}

/**
 * Intercepts browser-local utility commands. These commands never reach the
 * agent as user messages; help and skills produce anchored inline chat cards.
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
  const showHelp = useSlashCommandOutputStore((state) => state.showHelp);
  const showSkills = useSlashCommandOutputStore((state) => state.showSkills);

  return useCallback(
    (message: string) => {
      const command = message.trim();
      const isSystemCommand = [
        HELP_COMMAND,
        FEEDBACK_COMMAND,
        SKILLS_COMMAND,
        CONDENSE_COMMAND,
      ].includes(command);
      if (!isSystemCommand) {
        onSubmit(message);
        return;
      }

      if (command === FEEDBACK_COMMAND) {
        // @spec SC-004 — Feedback
        window.open(FEEDBACK_FORM_URL, "_blank", "noopener,noreferrer");
        return;
      }

      if (!conversationId) {
        displayErrorToast(
          t(I18nKey.SLASH_COMMAND$ACTIVE_CONVERSATION_REQUIRED),
        );
        return;
      }

      if (command === CONDENSE_COMMAND) {
        // @spec SC-005 — Conversation condensation
        condenseConversation(conversationId)
          .then(() =>
            displaySuccessToast(t(I18nKey.SLASH_COMMAND$CONDENSE_SUCCESS)),
          )
          .catch((error: unknown) => {
            const status = getHttpStatus(error);
            const message =
              status !== undefined &&
              CONDENSE_UNSUPPORTED_STATUS_CODES.has(status)
                ? t(I18nKey.SLASH_COMMAND$CONDENSE_UNSUPPORTED)
                : t(I18nKey.SLASH_COMMAND$CONDENSE_FAILED);
            displayErrorToast(message);
          });
        return;
      }

      const anchorEventId = getLastRenderableEventId();
      if (command === HELP_COMMAND) {
        // @spec SC-002 — Inline help
        refetchSkills()
          .then((result) => {
            showHelp(
              conversationId,
              anchorEventId,
              buildSlashCommandItems(
                result.isError ? (skills ?? []) : (result.data ?? skills ?? []),
              ),
            );
          })
          .catch(() => displayErrorToast(t(I18nKey.ERROR$GENERIC)));
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
