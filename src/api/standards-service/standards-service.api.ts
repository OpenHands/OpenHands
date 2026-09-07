import axios from "axios";
import { NoBackendAvailableError } from "../agent-server-client-options";
import { getEffectiveLocalBackend } from "../backend-registry/active-store";
import {
  SESSION_API_KEY_HEADER,
  STANDARDS_AUDIT_PATH,
  STANDARDS_CONFIG_PATH,
  STANDARDS_PLUGINS_PATH,
  STANDARDS_RUN_PATH,
} from "./standards-constants";
import type {
  StandardsAuditPage,
  StandardsAuditParams,
  StandardsConfig,
  StandardsPluginsResponse,
  StandardsRunRequest,
  StandardsRunResult,
} from "./standards-types";

const standardsAxios = axios.create();

standardsAxios.interceptors.request.use((config) => {
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

export const StandardsService = {
  listPlugins: async (root?: string): Promise<StandardsPluginsResponse> => {
    const { data } = await standardsAxios.get<StandardsPluginsResponse>(
      STANDARDS_PLUGINS_PATH,
      { params: root ? { root } : undefined },
    );
    return data;
  },

  getConfig: async (root?: string): Promise<StandardsConfig> => {
    const { data } = await standardsAxios.get<StandardsConfig>(
      STANDARDS_CONFIG_PATH,
      { params: root ? { root } : undefined },
    );
    return data;
  },

  putConfig: async (
    payload: Partial<StandardsConfig> & { root?: string },
  ): Promise<StandardsConfig> => {
    const { data } = await standardsAxios.put<StandardsConfig>(
      STANDARDS_CONFIG_PATH,
      payload,
    );
    return data;
  },

  run: async (payload: StandardsRunRequest): Promise<StandardsRunResult> => {
    const { data } = await standardsAxios.post<StandardsRunResult>(
      STANDARDS_RUN_PATH,
      payload,
    );
    return data;
  },

  getAudit: async (
    params: StandardsAuditParams = {},
  ): Promise<StandardsAuditPage> => {
    const { data } = await standardsAxios.get<StandardsAuditPage>(
      STANDARDS_AUDIT_PATH,
      { params },
    );
    return data;
  },
};

export default StandardsService;
