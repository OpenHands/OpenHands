import axios from "axios";
import { NoBackendAvailableError } from "../agent-server-client-options";
import { getEffectiveLocalBackend } from "../backend-registry/active-store";
import {
  FEATURE_DEV_API_PATH,
  SESSION_API_KEY_HEADER,
} from "./feature-developer-constants";
import type {
  FeatureDevReport,
  FeatureDevRun,
  StartFeatureDevRunPayload,
} from "./feature-developer-types";

const featureDevAxios = axios.create();

featureDevAxios.interceptors.request.use((config) => {
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

export const FeatureDeveloperService = {
  listRuns: async (filters?: {
    project_id?: string;
    status?: string;
  }): Promise<FeatureDevRun[]> => {
    const { data } = await featureDevAxios.get<FeatureDevRun[]>(
      FEATURE_DEV_API_PATH,
      { params: filters },
    );
    return data;
  },

  startRun: async (
    payload: StartFeatureDevRunPayload,
  ): Promise<FeatureDevRun> => {
    const { data } = await featureDevAxios.post<FeatureDevRun>(
      FEATURE_DEV_API_PATH,
      payload,
    );
    return data;
  },

  getRun: async (runId: string): Promise<FeatureDevRun> => {
    const { data } = await featureDevAxios.get<FeatureDevRun>(
      `${FEATURE_DEV_API_PATH}/${runId}`,
    );
    return data;
  },

  pauseRun: async (runId: string): Promise<FeatureDevRun> => {
    const { data } = await featureDevAxios.post<FeatureDevRun>(
      `${FEATURE_DEV_API_PATH}/${runId}/pause`,
    );
    return data;
  },

  resumeRun: async (runId: string): Promise<FeatureDevRun> => {
    const { data } = await featureDevAxios.post<FeatureDevRun>(
      `${FEATURE_DEV_API_PATH}/${runId}/resume`,
    );
    return data;
  },

  abortRun: async (runId: string): Promise<FeatureDevRun> => {
    const { data } = await featureDevAxios.post<FeatureDevRun>(
      `${FEATURE_DEV_API_PATH}/${runId}/abort`,
    );
    return data;
  },

  getReport: async (runId: string): Promise<FeatureDevReport> => {
    const { data } = await featureDevAxios.get<FeatureDevReport>(
      `${FEATURE_DEV_API_PATH}/${runId}/report`,
    );
    return data;
  },
};

export default FeatureDeveloperService;
