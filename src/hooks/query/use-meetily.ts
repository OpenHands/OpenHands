import { useMutation, useQueryClient } from "@tanstack/react-query";
import MeetilyService from "#/api/meetily-service/meetily-service.api";
import type { MeetilyTranscriptPayload } from "#/api/meetily-service/meetily-types";
import {
  KANBAN_QUERY_KEYS,
  MEETILY_QUERY_KEYS,
} from "#/hooks/query/query-keys";

export function usePreviewTranscript() {
  return useMutation({
    mutationKey: MEETILY_QUERY_KEYS.preview(),
    mutationFn: (payload: MeetilyTranscriptPayload) =>
      MeetilyService.preview(payload),
  });
}

export function useIngestTranscript() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: MeetilyTranscriptPayload) =>
      MeetilyService.ingest(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: KANBAN_QUERY_KEYS.all });
      queryClient.invalidateQueries({ queryKey: MEETILY_QUERY_KEYS.all });
    },
  });
}
