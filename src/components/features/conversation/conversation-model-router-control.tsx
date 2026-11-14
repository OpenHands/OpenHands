import React from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import type { ProfileInfo } from "@openhands/typescript-client";
import SparkleIcon from "#/icons/sparkle.svg?react";
import CheckIcon from "#/icons/checkmark.svg?react";
import { ContextMenu } from "#/ui/context-menu";
import { ContextMenuListItem } from "#/components/features/context-menu/context-menu-list-item";
import { Divider } from "#/ui/divider";
import { Typography } from "#/ui/typography";
import { ComboboxCaretInline } from "#/ui/combobox-caret";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
import { MODEL_ROUTER_QUERY_KEYS } from "#/hooks/query/query-keys";
import ModelRouterService, {
  type ModelRouterStatus,
} from "#/api/conversation-service/model-router-service.api";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

function routeKeyForProfile(
  profileName: string,
  status: ModelRouterStatus,
): string {
  const reserved = new Set(status.routes.map((route) => route.key));
  if (!status.enabled) reserved.add("default");
  if (!reserved.has(profileName)) return profileName;

  let suffix = 2;
  let candidate = `${profileName}-${suffix}`;
  while (reserved.has(candidate)) {
    suffix += 1;
    candidate = `${profileName}-${suffix}`;
  }
  return candidate;
}

function wasProfileAdded(
  profile: ProfileInfo,
  status: ModelRouterStatus,
): boolean {
  return status.routes.some(
    (route) =>
      route.key === profile.name || route.key.startsWith(`${profile.name}-`),
  );
}

function currentRouteKey(status: ModelRouterStatus): string | null {
  if (status.selection?.mode === "manual") {
    return status.selection.manual_route_key ?? null;
  }
  return status.last_route_key;
}

function routeDescription(route: { model?: string | null }) {
  return route.model?.trim() || null;
}

export function ConversationModelRouterControl() {
  const { t } = useTranslation("openhands");
  const { data: conversation } = useActiveConversation();
  const { data: profilesData } = useLlmProfiles();
  const { backend, orgId } = useActiveBackend();
  const queryClient = useQueryClient();
  const [isOpen, setIsOpen] = React.useState(false);
  const triggerRef = React.useRef<HTMLButtonElement>(null);
  const popoverRef = useClickOutsideElement<HTMLUListElement>(
    () => setIsOpen(false),
    triggerRef,
  );

  const conversationId = conversation?.id ?? null;
  const conversationUrl = conversation?.conversation_url ?? null;
  const sessionApiKey = conversation?.session_api_key ?? null;
  const canUseModelRouter = Boolean(
    conversationId &&
    conversation?.agent_kind !== "acp" &&
    !conversationId.startsWith("task-"),
  );
  const queryKey = conversationId
    ? MODEL_ROUTER_QUERY_KEYS.status(
        conversationId,
        backend.id,
        orgId,
        conversationUrl,
      )
    : null;

  // Do not include sessionApiKey in the query key; it is a credential.
  // eslint-disable-next-line @tanstack/query/exhaustive-deps
  const statusQuery = useQuery({
    queryKey: queryKey ?? ["model-router", "missing-conversation"],
    queryFn: () =>
      ModelRouterService.getModelRouter(
        conversationId!,
        conversationUrl,
        sessionApiKey,
      ),
    enabled: canUseModelRouter,
    retry: false,
    staleTime: 5000,
  });

  const updateCachedStatus = React.useCallback(
    (status: ModelRouterStatus) => {
      if (!queryKey) return;
      queryClient.setQueryData(queryKey, status);
    },
    [queryClient, queryKey],
  );

  const addRoute = useMutation({
    mutationFn: async (profileName: string) => {
      const status = statusQuery.data;
      if (!conversationId || !status) {
        throw new Error("Model router status is not loaded.");
      }
      return ModelRouterService.addProfileRoute(
        conversationId,
        conversationUrl,
        sessionApiKey,
        profileName,
        routeKeyForProfile(profileName, status),
      );
    },
    onSuccess: updateCachedStatus,
  });

  const switchRoute = useMutation({
    mutationFn: async (routeKey: string) => {
      if (!conversationId) throw new Error("Conversation is not loaded.");
      return ModelRouterService.switchRoute(
        conversationId,
        conversationUrl,
        sessionApiKey,
        routeKey,
      );
    },
    onSuccess: updateCachedStatus,
  });

  const resumeAutoRouting = useMutation({
    mutationFn: async () => {
      if (!conversationId) throw new Error("Conversation is not loaded.");
      return ModelRouterService.resumeAutoRouting(
        conversationId,
        conversationUrl,
        sessionApiKey,
      );
    },
    onSuccess: updateCachedStatus,
  });

  const status = statusQuery.data;
  if (!canUseModelRouter || statusQuery.isError || !status) return null;

  const profiles = profilesData?.profiles ?? [];
  const addableProfiles = profiles.filter(
    (profile) => !wasProfileAdded(profile, status),
  );
  const activeRouteKey = currentRouteKey(status);
  const activeRoute = status.routes.find(
    (route) => route.key === activeRouteKey,
  );
  const isMutating =
    addRoute.isPending || switchRoute.isPending || resumeAutoRouting.isPending;
  const mutationError =
    addRoute.error ?? switchRoute.error ?? resumeAutoRouting.error ?? null;
  const buttonLabel =
    activeRoute?.label ||
    activeRoute?.key ||
    conversation?.llm_model ||
    t(I18nKey.CONVERSATION_MODEL_ROUTER$BUTTON_LABEL);

  return (
    <div className="relative mr-2 shrink-0">
      <button
        ref={triggerRef}
        type="button"
        data-testid="conversation-model-router-control"
        aria-haspopup="dialog"
        aria-expanded={isOpen}
        aria-busy={isMutating}
        disabled={isMutating}
        title={t(I18nKey.CONVERSATION_MODEL_ROUTER$BUTTON_LABEL)}
        onClick={() => setIsOpen((open) => !open)}
        className={cn(
          "inline-flex h-7 max-w-56 items-center gap-1.5 rounded-md border border-border-subtle bg-surface px-2 text-xs text-foreground transition-colors",
          "hover:bg-interactive-hover disabled:cursor-not-allowed disabled:opacity-60",
        )}
      >
        <SparkleIcon className="size-3.5 shrink-0" aria-hidden />
        <span className="truncate">{buttonLabel}</span>
        <ComboboxCaretInline isOpen={isOpen} />
      </button>

      {isOpen && (
        <ContextMenu
          ref={popoverRef}
          testId="conversation-model-router-popover"
          position="bottom"
          alignment="right"
          className="right-0 mt-1 min-w-72 max-w-96 overflow-y-auto"
        >
          <li role="presentation" className="px-2 py-1">
            <Typography.Text className="menu-caption-text font-medium text-text-dim uppercase tracking-wide leading-4">
              {t(I18nKey.CONVERSATION_MODEL_ROUTER$MENU_TITLE)}
            </Typography.Text>
          </li>

          {status.enabled && (
            <>
              <ContextMenuListItem
                testId="conversation-model-router-auto-route"
                isDisabled={status.selection?.mode === "auto" || isMutating}
                onClick={(event) => {
                  event.preventDefault();
                  resumeAutoRouting.mutate();
                }}
                className="flex flex-col items-stretch gap-0.5"
              >
                <span className="flex items-center gap-2">
                  <span className="flex-1 truncate text-sm leading-5">
                    {t(I18nKey.CONVERSATION_MODEL_ROUTER$AUTO_ROUTING)}
                  </span>
                  {status.selection?.mode === "auto" && (
                    <CheckIcon className="size-3.5 shrink-0" aria-hidden />
                  )}
                </span>
                <span className="truncate text-xs leading-4 text-muted">
                  {t(
                    I18nKey.CONVERSATION_MODEL_ROUTER$AUTO_ROUTING_DESCRIPTION,
                  )}
                </span>
              </ContextMenuListItem>

              {status.routes.length > 0 && (
                <li role="presentation" className="px-2 pt-2 pb-0.5">
                  <Typography.Text className="menu-caption-text font-medium text-text-dim uppercase tracking-wide leading-4">
                    {t(I18nKey.CONVERSATION_MODEL_ROUTER$ROUTES)}
                  </Typography.Text>
                </li>
              )}

              {status.routes.map((route) => {
                const isActive = activeRouteKey === route.key;
                return (
                  <ContextMenuListItem
                    key={route.key}
                    testId={`conversation-model-router-route-${route.key}`}
                    isDisabled={isActive || isMutating}
                    onClick={(event) => {
                      event.preventDefault();
                      switchRoute.mutate(route.key);
                    }}
                    className={cn(
                      "flex flex-col items-stretch gap-0.5",
                      isActive && "bg-interactive-hover",
                    )}
                  >
                    <span className="flex items-center gap-2">
                      <span className="flex-1 truncate text-sm leading-5">
                        {route.label || route.key}
                      </span>
                      {isActive && (
                        <CheckIcon className="size-3.5 shrink-0" aria-hidden />
                      )}
                    </span>
                    {routeDescription(route) && (
                      <span className="truncate text-xs leading-4 text-muted">
                        {routeDescription(route)}
                      </span>
                    )}
                  </ContextMenuListItem>
                );
              })}
            </>
          )}

          <Divider inset="menu" />
          <li role="presentation" className="px-2 pt-1 pb-0.5">
            <Typography.Text className="menu-caption-text font-medium text-text-dim uppercase tracking-wide leading-4">
              {t(I18nKey.CONVERSATION_MODEL_ROUTER$ADD_PROFILE)}
            </Typography.Text>
          </li>

          {addableProfiles.length === 0 ? (
            <li className="px-2 py-1 text-xs text-muted">
              {t(I18nKey.CONVERSATION_MODEL_ROUTER$NO_PROFILES)}
            </li>
          ) : (
            addableProfiles.map((profile) => (
              <ContextMenuListItem
                key={profile.name}
                testId={`conversation-model-router-add-profile-${profile.name}`}
                isDisabled={isMutating}
                onClick={(event) => {
                  event.preventDefault();
                  addRoute.mutate(profile.name);
                }}
                className="flex flex-col items-stretch gap-0.5"
              >
                <span className="truncate text-sm leading-5">
                  {profile.name}
                </span>
                {profile.model && (
                  <span className="truncate text-xs leading-4 text-muted">
                    {profile.model}
                  </span>
                )}
              </ContextMenuListItem>
            ))
          )}

          {mutationError && (
            <li className="px-2 py-1 text-xs text-feedback-error">
              {t(I18nKey.CONVERSATION_MODEL_ROUTER$ERROR)}
            </li>
          )}
        </ContextMenu>
      )}
    </div>
  );
}
