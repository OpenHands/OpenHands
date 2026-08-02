import { useCallback, useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useNavigation } from "#/context/navigation-context";
import { getLastConversationTimelineEventId } from "#/hooks/chat/slash-command-timeline-boundary";
import { useForkConversation } from "#/hooks/mutation/use-fork-conversation";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useConversationSkills } from "#/hooks/query/use-conversation-skills";
import {
  SIDEBAR_RAIL_COLLAPSE_MAX_WIDTH,
  useBreakpoint,
} from "#/hooks/use-breakpoint";
import { I18nKey } from "#/i18n/declaration";
import { useOptionalSidebarMobileNav } from "#/components/features/sidebar/sidebar-mobile-nav-context";
import { useSidebarStore } from "#/stores/sidebar-store";
import { useSlashCommandOutputStore } from "#/stores/slash-command-output-store";
import {
  CONDENSE_COMMAND,
  FEEDBACK_COMMAND,
  CONFIRM_COMMAND,
  FORK_COMMAND,
  HELP_COMMAND,
  HISTORY_COMMAND,
  SETTINGS_COMMAND,
  SKILLS_COMMAND,
} from "#/utils/constants";
import {
  dismissToast,
  displayErrorToast,
  displayLoadingToast,
  displaySuccessToast,
  displayWarningToast,
} from "#/utils/custom-toast-handlers";
import { buildSlashCommandCatalog } from "#/utils/slash-command-catalog";
import { normalizeUiCommand } from "#/utils/slash-command-text";
import {
  PromiseDeadlineError,
  withPromiseDeadline,
} from "#/utils/promise-deadline";

export const OPENHANDS_FEEDBACK_URL = "https://forms.gle/chHc5VdS3wty5DwW6";
/** Slightly exceeds the Cloud client's 30-second transport timeout. */
export const SKILLS_COMMAND_DEADLINE_MS = 35_000;
export const HELP_COMMAND_DEADLINE_MS = 35_000;

interface UiCommandContext {
  outputScopeId: string | null | undefined;
  conversationId?: string | null;
  conversationUrl?: string | null;
  sessionApiKey?: string | null;
  agentKind?: "openhands" | "acp" | null;
  supportsManualCondensation?: boolean;
  onOpenConfirmationPolicy?: () => void;
  getTimelineBoundaryEventId?: () => string | null;
}

export const useUiCommandInterceptor = (
  onSubmit: (message: string) => void,
  {
    outputScopeId,
    conversationId = null,
    conversationUrl,
    sessionApiKey,
    agentKind,
    supportsManualCondensation,
    onOpenConfirmationPolicy,
    getTimelineBoundaryEventId,
  }: UiCommandContext,
) => {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();
  const isCloud = useActiveBackend().backend.kind === "cloud";
  const isMobileSidebar = useBreakpoint(SIDEBAR_RAIL_COLLAPSE_MAX_WIDTH);
  const toggleDesktopSidebar = useSidebarStore(
    (state) => state.toggleCollapsed,
  );
  const mobileSidebar = useOptionalSidebarMobileNav();
  const { data: skills, refetch: refetchSkills } = useConversationSkills();
  const beginSkills = useSlashCommandOutputStore((state) => state.beginSkills);
  const completeSkills = useSlashCommandOutputStore(
    (state) => state.completeSkills,
  );
  const failSkills = useSlashCommandOutputStore((state) => state.failSkills);
  const beginHelp = useSlashCommandOutputStore((state) => state.beginHelp);
  const completeHelp = useSlashCommandOutputStore(
    (state) => state.completeHelp,
  );
  const failHelp = useSlashCommandOutputStore((state) => state.failHelp);
  const deactivateSkillsPlacementFallback = useSlashCommandOutputStore(
    (state) => state.deactivateSkillsPlacementFallback,
  );
  const showSkills = useSlashCommandOutputStore((state) => state.showSkills);
  const showHelp = useSlashCommandOutputStore((state) => state.showHelp);
  const reserveInvocationOrder = useSlashCommandOutputStore(
    (state) => state.reserveInvocationOrder,
  );
  const condenseRequestsInFlight = useRef(new Set<string>());

  useEffect(
    () => () => {
      if (outputScopeId) deactivateSkillsPlacementFallback(outputScopeId);
    },
    [deactivateSkillsPlacementFallback, outputScopeId],
  );

  const showAvailableHelp = useCallback(
    (timelineBoundaryEventId: string | null, invocationOrder: number) => {
      const buildCatalog = (availableSkills: NonNullable<typeof skills>) =>
        buildSlashCommandCatalog({
          // Sparse skill records are still commands. Help and autocomplete
          // must consume the same catalog even without a description.
          skills: availableSkills,
          isCloud,
          hasConversation: !!conversationId,
          agentKind,
          supportsManualCondensation,
        });

      if (skills) {
        showHelp(
          outputScopeId!,
          timelineBoundaryEventId,
          buildCatalog(skills),
          invocationOrder,
        );
        return;
      }

      // Built-ins are frontend-owned, so accepting /help can always produce a
      // visible card immediately while optional skill enrichment is pending.
      const entryId = beginHelp(
        outputScopeId!,
        timelineBoundaryEventId,
        buildCatalog([]),
        invocationOrder,
      );
      withPromiseDeadline(
        Promise.resolve().then(() => refetchSkills({ throwOnError: true })),
        HELP_COMMAND_DEADLINE_MS,
        "Skill catalog request exceeded the command deadline.",
      ).then(
        (result) =>
          completeHelp(
            outputScopeId!,
            entryId,
            buildCatalog(result.data ?? []),
          ),
        (error: unknown) => {
          // Built-ins are frontend-owned and remain useful when optional skill
          // enrichment fails. The warning is separate from the help output.
          failHelp(outputScopeId!, entryId);
          displayErrorToast(
            error instanceof PromiseDeadlineError
              ? t(I18nKey.SLASH_COMMAND$RESOURCES_TIMEOUT)
              : error instanceof Error
                ? error.message
                : t(I18nKey.ERROR$GENERIC),
          );
        },
      );
    },
    [
      beginHelp,
      completeHelp,
      conversationId,
      agentKind,
      failHelp,
      isCloud,
      outputScopeId,
      refetchSkills,
      showHelp,
      skills,
      supportsManualCondensation,
      t,
    ],
  );

  return useCallback(
    (message: string) => {
      const command = normalizeUiCommand(message);

      if (command === HELP_COMMAND) {
        if (!outputScopeId) {
          displayErrorToast(t(I18nKey.SLASH_COMMAND$CONVERSATION_REQUIRED));
          return;
        }
        const timelineBoundaryEventId = getTimelineBoundaryEventId?.() ?? null;
        showAvailableHelp(timelineBoundaryEventId, reserveInvocationOrder());
        return;
      }
      if (command === CONFIRM_COMMAND) {
        if (isCloud) {
          displayWarningToast(
            t(I18nKey.SLASH_COMMAND$CONFIRM_CLOUD_UNSUPPORTED),
          );
          return;
        }
        if (!conversationId) {
          displayErrorToast(t(I18nKey.SLASH_COMMAND$CONVERSATION_REQUIRED));
          return;
        }
        if (agentKind !== "openhands" || !onOpenConfirmationPolicy) {
          displayErrorToast(
            t(I18nKey.SLASH_COMMAND$OPENHANDS_CONVERSATION_REQUIRED),
          );
          return;
        }
        onOpenConfirmationPolicy();
        return;
      }
      if (command === CONDENSE_COMMAND) {
        if (isCloud) {
          displayWarningToast(
            t(I18nKey.SLASH_COMMAND$CONDENSE_CLOUD_UNSUPPORTED),
          );
          return;
        }
        if (!conversationId) {
          displayErrorToast(t(I18nKey.SLASH_COMMAND$CONVERSATION_REQUIRED));
          return;
        }
        if (supportsManualCondensation !== true) {
          displayErrorToast(t(I18nKey.SLASH_COMMAND$CONDENSE_UNSUPPORTED));
          return;
        }
        if (condenseRequestsInFlight.current.has(conversationId)) return;

        const toastId = displayLoadingToast(
          t(I18nKey.SLASH_COMMAND$CONDENSE_PENDING),
        );
        condenseRequestsInFlight.current.add(conversationId);
        void (async () => {
          try {
            await AgentServerConversationService.condenseConversation(
              conversationId,
              conversationUrl,
              sessionApiKey,
            );
            dismissToast(toastId);
            displaySuccessToast(t(I18nKey.SLASH_COMMAND$CONDENSE_SUCCESS));
          } catch (error: unknown) {
            dismissToast(toastId);
            displayErrorToast(
              error instanceof Error ? error.message : t(I18nKey.ERROR$GENERIC),
            );
          } finally {
            condenseRequestsInFlight.current.delete(conversationId);
          }
        })();
        return;
      }
      if (command === SKILLS_COMMAND) {
        if (!outputScopeId) {
          displayErrorToast(t(I18nKey.SLASH_COMMAND$CONVERSATION_REQUIRED));
          return;
        }
        if (!conversationId) {
          // The home composer has no running conversation, so there cannot be
          // any conversation-loaded resources yet. Keep the CLI command
          // discoverable and render its canonical empty state locally.
          showSkills(
            outputScopeId,
            null,
            {
              skills: [],
              hooks: [],
              mcps: [],
            },
            reserveInvocationOrder(),
          );
          return;
        }
        const timelineBoundaryEventId = getTimelineBoundaryEventId?.() ?? null;
        const invocationOrder = reserveInvocationOrder();
        const entryId = beginSkills(
          outputScopeId,
          timelineBoundaryEventId,
          invocationOrder,
        );
        withPromiseDeadline(
          AgentServerConversationService.getLoadedResources(
            conversationId,
            conversationUrl,
            sessionApiKey,
          ),
          SKILLS_COMMAND_DEADLINE_MS,
          "Loaded-resource request exceeded the command deadline.",
        ).then(
          (resources) => completeSkills(outputScopeId, entryId, resources),
          (error: unknown) =>
            failSkills(
              outputScopeId,
              entryId,
              error instanceof PromiseDeadlineError ? "timeout" : "request",
            ),
        );
        return;
      }
      if (command === HISTORY_COMMAND) {
        if (isMobileSidebar && mobileSidebar) mobileSidebar.toggle();
        else toggleDesktopSidebar();
        return;
      }
      if (command === SETTINGS_COMMAND) {
        navigate("/settings");
        return;
      }
      if (command === FEEDBACK_COMMAND) {
        window.open(OPENHANDS_FEEDBACK_URL, "_blank", "noopener,noreferrer");
        return;
      }
      onSubmit(message);
    },
    [
      beginSkills,
      beginHelp,
      agentKind,
      completeSkills,
      completeHelp,
      conversationId,
      conversationUrl,
      failSkills,
      failHelp,
      getTimelineBoundaryEventId,
      isCloud,
      isMobileSidebar,
      mobileSidebar,
      navigate,
      onSubmit,
      onOpenConfirmationPolicy,
      outputScopeId,
      reserveInvocationOrder,
      showHelp,
      showSkills,
      sessionApiKey,
      t,
      toggleDesktopSidebar,
      showAvailableHelp,
      supportsManualCondensation,
    ],
  );
};

export const useCliCommandInterceptor = (
  conversationId: string | null | undefined,
  onSubmit: (message: string) => void,
  options?: { onOpenConfirmationPolicy?: () => void },
) => {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();
  const isCloud = useActiveBackend().backend.kind === "cloud";
  const { data: conversation } = useActiveConversation();
  const { mutate: forkConversation } = useForkConversation();

  const handleConversationCommand = useCallback(
    (message: string) => {
      const command = normalizeUiCommand(message);

      if (command === FORK_COMMAND) {
        if (!conversationId || isCloud) {
          displayErrorToast(
            t(I18nKey.SLASH_COMMAND$LOCAL_CONVERSATION_REQUIRED),
          );
          return;
        }
        const sourceTitle =
          conversation?.id === conversationId ? conversation.title : undefined;
        forkConversation(
          {
            sourceConversationId: conversationId,
            ...(sourceTitle
              ? {
                  title: t(I18nKey.CONVERSATION$FORK_TITLE, {
                    title: sourceTitle,
                  }),
                }
              : {}),
          },
          {
            onSuccess: ({ info }) => navigate(`/conversations/${info.id}`),
            onError: (error) =>
              displayErrorToast(
                error instanceof Error
                  ? error.message
                  : t(I18nKey.ERROR$GENERIC),
              ),
          },
        );
        return;
      }
      onSubmit(message);
    },
    [
      conversationId,
      conversation,
      forkConversation,
      isCloud,
      navigate,
      onSubmit,
      t,
    ],
  );

  return useUiCommandInterceptor(handleConversationCommand, {
    outputScopeId: conversationId,
    conversationId,
    conversationUrl: conversation?.conversation_url,
    sessionApiKey: conversation?.session_api_key,
    agentKind:
      conversation && conversation.id === conversationId
        ? conversation.agent_kind
        : null,
    supportsManualCondensation:
      conversation && conversation.id === conversationId
        ? conversation.supports_manual_condensation
        : undefined,
    onOpenConfirmationPolicy: options?.onOpenConfirmationPolicy,
    getTimelineBoundaryEventId: getLastConversationTimelineEventId,
  });
};
