import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import ChannelService from "#/api/channel-service/channel-service.api";
import type {
  ChannelConfigPayload,
  ListMessagesFilters,
} from "#/api/channel-service/channel-types";
import { CHANNELS_QUERY_KEYS } from "#/hooks/query/query-keys";

export function useChannels() {
  return useQuery({
    queryKey: CHANNELS_QUERY_KEYS.list(),
    queryFn: () => ChannelService.list(),
  });
}

export function useChannelMessages(filters?: ListMessagesFilters) {
  return useQuery({
    queryKey: [...CHANNELS_QUERY_KEYS.messages(), filters] as const,
    queryFn: () => ChannelService.listMessages(filters),
  });
}

function useInvalidateChannels() {
  const queryClient = useQueryClient();
  return () => {
    queryClient.invalidateQueries({ queryKey: CHANNELS_QUERY_KEYS.all });
  };
}

export function useStartChannel() {
  const invalidate = useInvalidateChannels();
  return useMutation({
    mutationFn: (channelId: string) => ChannelService.start(channelId),
    onSuccess: invalidate,
  });
}

export function useStopChannel() {
  const invalidate = useInvalidateChannels();
  return useMutation({
    mutationFn: (channelId: string) => ChannelService.stop(channelId),
    onSuccess: invalidate,
  });
}

export function useUpdateChannelConfig() {
  const invalidate = useInvalidateChannels();
  return useMutation({
    mutationFn: ({
      channelId,
      payload,
    }: {
      channelId: string;
      payload: ChannelConfigPayload;
    }) => ChannelService.updateConfig(channelId, payload),
    onSuccess: invalidate,
  });
}
