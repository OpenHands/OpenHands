/**
 * Mid-conversation autonomy PATCH + local metadata sync.
 * @spec PROJETOSIN-195
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useState } from "react";
import EngagementService from "#/api/pentest/engagement-service";
import type { AutonomyPropagation } from "#/api/pentest/engagement-types";
import {
  getStoredConversationMetadata,
  setStoredConversationMetadata,
} from "#/api/conversation-metadata-store";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { PENTEST_ENGAGEMENTS_QUERY_KEYS } from "#/hooks/query/query-keys";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import type { AutonomyMode } from "#/types/workspace-types";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";

export function useEngagementAutonomy(options: {
  engagementId: string | null;
  conversationId: string | null;
  enabled?: boolean;
}) {
  const { engagementId, conversationId, enabled = true } = options;
  const { backend, orgId } = useActiveBackend();
  const queryClient = useQueryClient();
  const { t } = useTranslation("openhands");
  const [propagation, setPropagation] = useState<AutonomyPropagation | null>(
    null,
  );
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const query = useQuery({
    queryKey: engagementId
      ? [
          ...PENTEST_ENGAGEMENTS_QUERY_KEYS.detail(engagementId),
          backend.id,
          orgId,
        ]
      : ["pentest-engagements", "detail", "none"],
    queryFn: () => EngagementService.getEngagement(engagementId!),
    enabled: Boolean(engagementId) && enabled,
    retry: false,
    meta: { disableToast: true },
  });

  const mutation = useMutation({
    mutationFn: (mode: AutonomyMode) =>
      EngagementService.patchAutonomyMode(engagementId!, mode),
    onSuccess: (detail) => {
      setErrorMessage(null);
      setPropagation(detail.propagation ?? null);
      queryClient.setQueryData(
        [
          ...PENTEST_ENGAGEMENTS_QUERY_KEYS.detail(engagementId!),
          backend.id,
          orgId,
        ],
        detail,
      );
      if (conversationId) {
        const prev = getStoredConversationMetadata(conversationId);
        if (prev) {
          setStoredConversationMetadata(conversationId, {
            ...prev,
            autonomy_mode: detail.autonomy_mode,
          });
        }
      }
      displaySuccessToast(t(I18nKey.PENTEST$AUTONOMY_SAVE_SUCCESS));
    },
    onError: () => {
      const message = t(I18nKey.PENTEST$AUTONOMY_SAVE_ERROR);
      setErrorMessage(message);
      displayErrorToast(message);
    },
  });

  const { mutateAsync, isPending } = mutation;

  const patchAutonomy = useCallback(
    async (mode: AutonomyMode) => {
      if (!engagementId) return;
      await mutateAsync(mode);
    },
    [engagementId, mutateAsync],
  );

  const metadataMode = conversationId
    ? getStoredConversationMetadata(conversationId)?.autonomy_mode
    : null;

  const autonomyMode: AutonomyMode =
    query.data?.autonomy_mode ?? metadataMode ?? "semi_autonomous";

  const isArchived = query.data?.status === "archived";

  return {
    autonomyMode,
    isLoading: query.isLoading,
    isSaving: isPending,
    isArchived,
    propagation,
    errorMessage,
    patchAutonomy,
    clearError: () => setErrorMessage(null),
  };
}
