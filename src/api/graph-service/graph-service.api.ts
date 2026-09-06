import axios from "axios";
import { NoBackendAvailableError } from "../agent-server-client-options";
import { getEffectiveLocalBackend } from "../backend-registry/active-store";
import {
  GRAPH_CONFIG_PATH,
  GRAPH_DEFINITIONS_PATH,
  GRAPH_IMPORT_PATH,
  GRAPH_INDEX_PATH,
  GRAPH_QUERY_PATH,
  GRAPH_RETRIGGER_PATH,
  GRAPH_STATUS_PATH,
  SESSION_API_KEY_HEADER,
} from "./graph-constants";
import type {
  GraphConfig,
  GraphIndexRequest,
  GraphIndexStatus,
  GraphQueryParams,
  GraphQueryResult,
} from "./graph-types";

const graphAxios = axios.create();

graphAxios.interceptors.request.use((config) => {
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

export const GraphService = {
  index: async (payload: GraphIndexRequest = {}): Promise<GraphIndexStatus> => {
    const { data } = await graphAxios.post<GraphIndexStatus>(
      GRAPH_INDEX_PATH,
      payload,
    );
    return data;
  },

  getStatus: async (root?: string): Promise<GraphIndexStatus> => {
    const { data } = await graphAxios.get<GraphIndexStatus>(GRAPH_STATUS_PATH, {
      params: root ? { root } : undefined,
    });
    return data;
  },

  clearIndex: async (root?: string): Promise<GraphIndexStatus> => {
    const { data } = await graphAxios.delete<GraphIndexStatus>(
      GRAPH_INDEX_PATH,
      { data: root ? { root } : {} },
    );
    return data;
  },

  retrigger: async (root?: string): Promise<GraphIndexStatus> => {
    const { data } = await graphAxios.post<GraphIndexStatus>(
      GRAPH_RETRIGGER_PATH,
      root ? { root } : {},
    );
    return data;
  },

  query: async (params: GraphQueryParams): Promise<GraphQueryResult> => {
    const { data } = await graphAxios.get<GraphQueryResult>(GRAPH_QUERY_PATH, {
      params,
    });
    return data;
  },

  definitions: async (
    symbol: string,
    root?: string,
  ): Promise<GraphQueryResult> => {
    const { data } = await graphAxios.get<GraphQueryResult>(
      GRAPH_DEFINITIONS_PATH,
      { params: { symbol, root } },
    );
    return data;
  },

  getConfig: async (): Promise<GraphConfig> => {
    const { data } = await graphAxios.get<GraphConfig>(GRAPH_CONFIG_PATH);
    return data;
  },

  putConfig: async (payload: Partial<GraphConfig>): Promise<GraphConfig> => {
    const { data } = await graphAxios.put<GraphConfig>(
      GRAPH_CONFIG_PATH,
      payload,
    );
    return data;
  },

  importProjectConfig: async (path: string): Promise<GraphConfig> => {
    const { data } = await graphAxios.post<GraphConfig>(GRAPH_IMPORT_PATH, {
      path,
    });
    return data;
  },
};

export default GraphService;
