import { useMutation, useQueryClient } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { DirectConversationInfo } from "#/api/agent-server-adapter";
import { useNavigation } from "#/context/navigation-context";
import { setConversationState } from "#/utils/conversation-local-storage";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

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
 * Completion side effects (navigation, draft restoration, error toast) run in
 * the mutation-level callbacks, not per-call ones: a virtualized chat can
 * unmount the message row — and with it the per-call `onSuccess` — while the
 * fork request is still in flight, which would strand the user on the source
 * conversation. Mutation callbacks live on the cache entry and fire even when
 * the calling component has unmounted.
 *
 * Because the callbacks outlive the row, navigation always runs once the fork
 * lands, even if the reader has since moved to another conversation. That is
 * deliberate: the fork is the result of an explicit action, and the row can
 * unmount (virtualization) while it is in flight, so there is no reliable
 * per-row "am I still mounted?" signal to fall back on.
 */
export const useForkConversation = (sourceConversationId: string) => {
  const queryClient = useQueryClient();
  const { navigate } = useNavigation();

  return useMutation({
    mutationKey: forkConversationMutationKey(sourceConversationId),
    mutationFn: async ({
      eventId,
      editText,
      title,
    }: ForkConversationVariables): Promise<ForkConversationResult> => {
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
      // Surface the fork in the sidebar without wiping loaded pages.
      queryClient.invalidateQueries({ queryKey: ["user", "conversations"] });
      navigate(`/conversations/${info.id}`);
    },
    onError: (error) =>
      displayErrorToast(error instanceof Error ? error.message : null),
  });
};
