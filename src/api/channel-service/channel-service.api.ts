import axios from "axios";
import { NoBackendAvailableError } from "../agent-server-client-options";
import { getEffectiveLocalBackend } from "../backend-registry/active-store";
import {
  CHANNELS_API_PATH,
  CHANNELS_MESSAGES_PATH,
  SESSION_API_KEY_HEADER,
  channelConfigPath,
  channelDetailPath,
  channelStartPath,
  channelStopPath,
} from "./channel-constants";
import type {
  ChannelConfigPayload,
  ChannelMessagePage,
  ChannelRecord,
  ListMessagesFilters,
} from "./channel-types";

const channelAxios = axios.create();

channelAxios.interceptors.request.use((config) => {
  const backend = getEffectiveLocalBackend();
  if (!backend) throw new NoBackendAvailableError();
  // eslint-disable-next-line no-param-reassign
  config.baseURL = backend.host;
  const apiKey = backend.apiKey?.trim();
  if (apiKey) {
    config.headers.set(SESSION_API_KEY_HEADER, apiKey);
  }
  return config;
});

export const ChannelService = {
  list: async (): Promise<ChannelRecord[]> => {
    const { data } = await channelAxios.get<ChannelRecord[]>(CHANNELS_API_PATH);
    return data;
  },

  get: async (channelId: string): Promise<ChannelRecord> => {
    const { data } = await channelAxios.get<ChannelRecord>(
      channelDetailPath(channelId),
    );
    return data;
  },

  start: async (channelId: string): Promise<ChannelRecord> => {
    const { data } = await channelAxios.post<ChannelRecord>(
      channelStartPath(channelId),
    );
    return data;
  },

  stop: async (channelId: string): Promise<ChannelRecord> => {
    const { data } = await channelAxios.post<ChannelRecord>(
      channelStopPath(channelId),
    );
    return data;
  },

  updateConfig: async (
    channelId: string,
    payload: ChannelConfigPayload,
  ): Promise<ChannelRecord> => {
    const { data } = await channelAxios.put<ChannelRecord>(
      channelConfigPath(channelId),
      payload,
    );
    return data;
  },

  listMessages: async (
    filters?: ListMessagesFilters,
  ): Promise<ChannelMessagePage> => {
    const { data } = await channelAxios.get<ChannelMessagePage>(
      CHANNELS_MESSAGES_PATH,
      { params: filters },
    );
    return data;
  },
};

export default ChannelService;
