import { useMutation, useQueryClient } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { DirectConversationInfo } from "#/api/agent-server-adapter";
import { useNavigation } from "#/context/navigation-context";
import { setConversationState } from "#/utils/conversation-local-storage";
import i18n from "#/i18n";
import { I18nKey } from "#/i18n/declaration";
import { getAgentCanvasBasePath } from "#/utils/base-path";
import {
  displayErrorToast,
  displaySuccessToastWithLink,
} from "#/utils/custom-toast-handlers";

/** The app-relative path of a conversation's route. */
const conversationPath = (conversationId: string) =>
  `/conversations/${conversationId}`;

/**
 * The active app-relative route, read from the navigation owner rather than the
 * document URL. Prefers the owner's live getter (correct even after the caller
 * unmounts and when an embedding host routes without updating the URL), then
 * the route the owner last supplied, and only falls back to the document URL
 * for a provider-less context. `window.location` alone is wrong when an
 * embedding host drives routing without changing it.
 */
const resolveCurrentAppPath = (
  getCurrentPath?: () => string,
  ownerPath?: string | null,
): string => {
  const live = getCurrentPath?.();
  if (typeof live === "string") return live;
  if (typeof ownerPath === "string") return ownerPath;
  if (typeof window === "undefined") return "";
  const base = getAgentCanvasBasePath();
  const { pathname } = window.location;
  return base && pathname.startsWith(base)
    ? pathname.slice(base.length)
    : pathname;
};

/**
 * Whether `path` is the conversation the fork branched from (or a sub-route of
 * it, e.g. its panel).
 */
const isViewingConversation = (
  sourceConversationId: string,
  path: string,
): boolean => {
  const source = conversationPath(sourceConversationId);
  return path === source || path.startsWith(`${source}/`);
};

/**
 * Mutation key prefix for {@link useForkConversation}. Callers read the
 * in-flight state with `useIsMutating`, which lives on the cache entry and
 * therefore survives a caller unmounting (a virtualized row scrolling out).
 */
export const FORK_CONVERSATION_MUTATION_KEY = ["fork-conversation"];

/**
 * Mutation key for a fork of one specific conversation. The key must be scoped
 * to the source conversation: an unscoped `useIsMutating` matches a fork of
 * *any* conversation, so while a fork from conversation A is in flight a
 * `Branch` click in conversation B would read as "already forking" and silently
 * do nothing.
 */
export const forkConversationMutationKey = (
  sourceConversationId: string,
): string[] => [...FORK_CONVERSATION_MUTATION_KEY, sourceConversationId];

interface ForkConversationVariables {
  /** The message the action was invoked on. */
  eventId: string;
  /** Set for an "edit message" branch: exclude the message and restore this text. */
  editText?: string | null;
  /** Optional title for the fork, so it reads distinctly from its source. */
  title?: string;
}

interface ForkConversationResult {
  info: DirectConversationInfo;
}

/**
 * Branches a conversation from a message. Edit-mode (`editText` set) resolves
 * the message's parent (via getEventParentId) and branches there, excluding
 * the message; otherwise it branches at the message (inclusive). Local
 * agent-server only.
 *
 * Completion side effects (draft restoration, navigation, error toast) run in
 * the mutation-level callbacks, not per-call ones: a virtualized chat can
 * unmount the message row — and with it the per-call `onSuccess` — while the
 * fork request is still in flight, which would strand the user on the source
 * conversation. Mutation callbacks live on the cache entry and fire even when
 * the calling component has unmounted.
 *
 * Because the callbacks outlive the row, they must not assume the reader is
 * still looking at the source conversation. The reader may have navigated
 * elsewhere while the fork was in flight, and pulling them into the fork would
 * hijack that navigation. Navigation therefore runs only when the active route
 * — read from the navigation owner, so it is right for an embedded host that
 * routes without changing the document URL — is still the source conversation;
 * otherwise the fork is announced with a success toast whose link routes back
 * through that owner, and the reader stays put. The saved draft is written
 * unconditionally, so opening the fork from the toast still restores the text.
 */
export const useForkConversation = (sourceConversationId: string) => {
  const queryClient = useQueryClient();
  const { navigate, getCurrentPath, currentPath } = useNavigation();

  return useMutation({
    mutationKey: forkConversationMutationKey(sourceConversationId),
    mutationFn: async ({
      eventId,
      editText,
      title,
    }: ForkConversationVariables): Promise<ForkConversationResult> => {
      if (!sourceConversationId) {
        // A fork needs a source to branch from and a route to open. The caller's
        // Branch action is already gated on a conversation, so this only guards
        // a stale render; fail fast rather than registering a fork under the
        // shared `["fork-conversation", ""]` key, which every id-less view would
        // then match. `onError` forwards the message straight to the toast, so
        // it has to be translated copy, not an English literal.
        throw new Error(i18n.t(I18nKey.CONVERSATION$BRANCH_MISSING_SOURCE));
      }
      let fromEventId = eventId;
      let excluded = false;

      if (editText != null) {
        const parentId = await AgentServerConversationService.getEventParentId(
          sourceConversationId,
          eventId,
        );
        if (parentId) {
          fromEventId = parentId;
          excluded = true;
        }
      }

      const info = await AgentServerConversationService.forkConversation(
        sourceConversationId,
        fromEventId,
        title,
      );

      // Older agent-servers (< 1.31.0) ignore `from_event_id` and copy the
      // whole conversation. When honored, the fork's HEAD is `fromEventId`; if
      // not, the message wasn't excluded — don't prefill (would duplicate).
      if (excluded) {
        const leafEventId = (info as { leaf_event_id?: string | null })
          .leaf_event_id;
        if (leafEventId !== fromEventId) {
          excluded = false;
        }
      }

      // Seed the fork's saved draft before navigating: the composer clears
      // itself on the conversation switch and then restores that draft, so
      // text pushed into it before the route settles would be wiped.
      if (excluded && editText) {
        setConversationState(info.id, { draftMessage: editText });
      }

      return { info };
    },
    onSuccess: ({ info }) => {
      // Guard a stale settle: with no source there is no route to compare
      // against and no conversation to branch from (mutationFn already throws).
      if (!sourceConversationId) return;
      // Surface the fork in the sidebar without wiping loaded pages.
      queryClient.invalidateQueries({ queryKey: ["user", "conversations"] });
      const forkPath = conversationPath(info.id);
      // Read the route from the navigation owner, not `window.location`: an
      // embedding host can route Canvas without touching the document URL, so
      // the document path would report the reader as having left the source
      // when they are still viewing it.
      const stillViewingSource = isViewingConversation(
        sourceConversationId,
        resolveCurrentAppPath(getCurrentPath, currentPath),
      );
      if (stillViewingSource) {
        navigate(forkPath);
      } else {
        // The reader moved on while the fork was in flight. Announce it with a
        // link instead of dragging them out of where they are. The link routes
        // through the navigation owner so an embedded Canvas stays embedded.
        displaySuccessToastWithLink(
          i18n.t(I18nKey.CONVERSATION$BRANCHED),
          i18n.t(I18nKey.CHILD_CONVERSATION$OPEN),
          forkPath,
          () => navigate(forkPath),
        );
      }
    },
    onError: (error) =>
      displayErrorToast(error instanceof Error ? error.message : null),
  });
};
