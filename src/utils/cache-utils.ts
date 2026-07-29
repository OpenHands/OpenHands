import { QueryClient } from "@tanstack/react-query";
import type { ActionEvent } from "#/types/agent-server/core/events/action-event";
import { useModelStore } from "#/stores/model-store";
import { stripWorkspacePrefix } from "./path-utils";

/**
 * Cache invalidation utilities for TanStack Query
 */

/**
 * Handle cache invalidation for ActionEvent
 * Invalidates relevant query caches based on the action type
 *
 * @param event - The ActionEvent to process
 * @param conversationId - The conversation ID for cache keys
 * @param queryClient - The TanStack Query client instance
 */
export const handleActionEventCacheInvalidation = (
  event: ActionEvent,
  conversationId: string,
  queryClient: QueryClient,
) => {
  const { action } = event;

  // Invalidate file_changes cache for file-related actions
  if (
    action.kind === "StrReplaceEditorAction" ||
    action.kind === "FileEditorAction" ||
    action.kind === "ExecuteBashAction"
  ) {
    queryClient.invalidateQueries(
      {
        queryKey: ["file_changes", conversationId],
      },
      { cancelRefetch: false },
    );
  }

  // Invalidate specific file diff cache for file modifications
  if (
    (action.kind === "StrReplaceEditorAction" ||
      action.kind === "FileEditorAction") &&
    action.path
  ) {
    const strippedPath = stripWorkspacePrefix(action.path);
    queryClient.invalidateQueries({
      queryKey: ["file_diff", conversationId, strippedPath],
    });
  }

  // When the agent autonomously swaps the LLM via SwitchLLMTool, refresh the
  // conversation so SwitchProfileButton / ChatInputModel pick up the new
  // `llm_model`, and drop any previously-set optimistic profile name (it
  // would now misrepresent the agent's choice).
  if (event.tool_name === "SwitchLLMTool") {
    queryClient.invalidateQueries({
      queryKey: ["user", "conversation", conversationId],
    });
    useModelStore.getState().clearActiveProfile(conversationId);
  }
};
