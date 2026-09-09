import axios from "axios";
import { NoBackendAvailableError } from "../agent-server-client-options";
import { getEffectiveLocalBackend } from "../backend-registry/active-store";
import {
  MEETINGS_ACTION_PATH,
  MEETINGS_TRANSCRIPT_PATH,
  SESSION_API_KEY_HEADER,
} from "./meetily-constants";
import type {
  MeetilyIngestResult,
  MeetilyPreview,
  MeetilyTranscriptPayload,
} from "./meetily-types";

const meetilyAxios = axios.create();

meetilyAxios.interceptors.request.use((config) => {
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

export const MeetilyService = {
  preview: async (
    payload: MeetilyTranscriptPayload,
  ): Promise<MeetilyPreview> => {
    const { data } = await meetilyAxios.post<MeetilyPreview>(
      MEETINGS_ACTION_PATH,
      payload,
    );
    return data;
  },

  ingest: async (
    payload: MeetilyTranscriptPayload,
  ): Promise<MeetilyIngestResult> => {
    const { data } = await meetilyAxios.post<MeetilyIngestResult>(
      MEETINGS_TRANSCRIPT_PATH,
      payload,
    );
    return data;
  },
};

export default MeetilyService;
