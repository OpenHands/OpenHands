import axios from "axios";
import { NoBackendAvailableError } from "../agent-server-client-options";
import { getEffectiveLocalBackend } from "../backend-registry/active-store";
import {
  ROUTING_AUDIT_PATH,
  ROUTING_CONFIG_PATH,
  ROUTING_INGEST_PATH,
  ROUTING_LOCAL_RUNTIMES_PATH,
  ROUTING_REGISTRY_PATH,
  ROUTING_RESOLVE_PATH,
  ROUTING_ROUTER_MODEL_PATH,
  ROUTING_SOURCES_PATH,
  ROUTING_TAXONOMY_PATH,
  ROUTING_PRIVACY_REFRESH_PATH,
  ROUTING_IMPORT_PATH,
  SESSION_API_KEY_HEADER,
} from "./routing-constants";
import type {
  RoutingAuditPage,
  RoutingConfig,
  RoutingIngestResult,
  RoutingLocalRuntimes,
  RoutingRegistrySnapshot,
  RoutingResolveRequest,
  RoutingResolveResult,
  RoutingRouterModelConfig,
  RoutingRouterModelResponse,
  RoutingSourcesResponse,
  RoutingTaxonomy,
} from "./routing-types";

const routingAxios = axios.create();

routingAxios.interceptors.request.use((config) => {
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

export const RoutingService = {
  getConfig: async (): Promise<RoutingConfig> => {
    const { data } = await routingAxios.get<RoutingConfig>(ROUTING_CONFIG_PATH);
    return data;
  },

  putConfig: async (
    payload: Partial<RoutingConfig>,
  ): Promise<RoutingConfig> => {
    const { data } = await routingAxios.put<RoutingConfig>(
      ROUTING_CONFIG_PATH,
      payload,
    );
    return data;
  },

  importProjectConfig: async (path: string): Promise<RoutingConfig> => {
    const { data } = await routingAxios.post<RoutingConfig>(
      ROUTING_IMPORT_PATH,
      {
        path,
      },
    );
    return data;
  },

  getTaxonomy: async (): Promise<RoutingTaxonomy> => {
    const { data } = await routingAxios.get<RoutingTaxonomy>(
      ROUTING_TAXONOMY_PATH,
    );
    return data;
  },

  putTaxonomy: async (
    payload: Partial<RoutingTaxonomy> & { reset?: boolean },
  ): Promise<RoutingTaxonomy> => {
    const { data } = await routingAxios.put<RoutingTaxonomy>(
      ROUTING_TAXONOMY_PATH,
      payload,
    );
    return data;
  },

  getRegistry: async (
    connectedProviders?: string[],
  ): Promise<RoutingRegistrySnapshot> => {
    const { data } = await routingAxios.get<RoutingRegistrySnapshot>(
      ROUTING_REGISTRY_PATH,
      {
        params: connectedProviders
          ? { connected_providers: connectedProviders.join(",") }
          : undefined,
      },
    );
    return data;
  },

  ingestBenchmarks: async (
    sources?: string[],
  ): Promise<RoutingIngestResult> => {
    const { data } = await routingAxios.post<RoutingIngestResult>(
      ROUTING_INGEST_PATH,
      sources ? { sources } : {},
    );
    return data;
  },

  getSources: async (): Promise<RoutingSourcesResponse> => {
    const { data } =
      await routingAxios.get<RoutingSourcesResponse>(ROUTING_SOURCES_PATH);
    return data;
  },

  refreshPrivacy: async (payload: unknown): Promise<unknown> => {
    const { data } = await routingAxios.post(
      ROUTING_PRIVACY_REFRESH_PATH,
      payload,
    );
    return data;
  },

  getRouterModel: async (
    preset?: string,
    connectedProviders?: string[],
  ): Promise<RoutingRouterModelResponse> => {
    const { data } = await routingAxios.get<RoutingRouterModelResponse>(
      ROUTING_ROUTER_MODEL_PATH,
      {
        params: {
          ...(preset ? { preset } : {}),
          ...(connectedProviders
            ? { connected_providers: connectedProviders.join(",") }
            : {}),
        },
      },
    );
    return data;
  },

  putRouterModel: async (
    payload: Partial<RoutingRouterModelConfig>,
  ): Promise<RoutingRouterModelResponse> => {
    const { data } = await routingAxios.put<RoutingRouterModelResponse>(
      ROUTING_ROUTER_MODEL_PATH,
      payload,
    );
    return data;
  },

  getLocalRuntimes: async (): Promise<RoutingLocalRuntimes> => {
    const { data } = await routingAxios.get<RoutingLocalRuntimes>(
      ROUTING_LOCAL_RUNTIMES_PATH,
    );
    return data;
  },

  resolve: async (
    payload: RoutingResolveRequest,
  ): Promise<RoutingResolveResult> => {
    const { data } = await routingAxios.post<RoutingResolveResult>(
      ROUTING_RESOLVE_PATH,
      payload,
    );
    return data;
  },

  getAudit: async (filters?: {
    card_id?: string;
    run_id?: string;
    kind?: string;
    limit?: number;
    offset?: number;
  }): Promise<RoutingAuditPage> => {
    const { data } = await routingAxios.get<RoutingAuditPage>(
      ROUTING_AUDIT_PATH,
      { params: filters },
    );
    return data;
  },
};

export default RoutingService;
