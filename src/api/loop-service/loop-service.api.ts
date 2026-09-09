import axios from "axios";
import { NoBackendAvailableError } from "../agent-server-client-options";
import { getEffectiveLocalBackend } from "../backend-registry/active-store";
import {
  LOOPS_API_PATH,
  LOOPS_TRIGGERS_API_PATH,
  SESSION_API_KEY_HEADER,
} from "./loop-constants";
import type {
  CreateLoopTriggerPayload,
  FireTriggerResult,
  LoopDefinition,
  LoopRun,
  LoopTrigger,
  TriggerEvent,
  UpdateLoopTriggerPayload,
} from "./loop-types";

const loopAxios = axios.create();

loopAxios.interceptors.request.use((config) => {
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

export const LoopService = {
  listDefinitions: async (): Promise<LoopDefinition[]> => {
    const { data } = await loopAxios.get<LoopDefinition[]>(LOOPS_API_PATH);
    return data;
  },

  listRuns: async (definitionId: string): Promise<LoopRun[]> => {
    const { data } = await loopAxios.get<LoopRun[]>(
      `${LOOPS_API_PATH}/${definitionId}/runs`,
    );
    return data;
  },

  getRun: async (runId: string): Promise<LoopRun> => {
    const { data } = await loopAxios.get<LoopRun>(
      `${LOOPS_API_PATH}/runs/${runId}`,
    );
    return data;
  },

  listTriggers: async (filters?: {
    project_id?: string;
    trigger_type?: string;
    enabled?: boolean;
  }): Promise<LoopTrigger[]> => {
    const { data } = await loopAxios.get<LoopTrigger[]>(
      LOOPS_TRIGGERS_API_PATH,
      { params: filters },
    );
    return data;
  },

  createTrigger: async (
    payload: CreateLoopTriggerPayload,
  ): Promise<LoopTrigger> => {
    const { data } = await loopAxios.post<LoopTrigger>(
      LOOPS_TRIGGERS_API_PATH,
      payload,
    );
    return data;
  },

  updateTrigger: async (
    triggerId: string,
    payload: UpdateLoopTriggerPayload,
  ): Promise<LoopTrigger> => {
    const { data } = await loopAxios.patch<LoopTrigger>(
      `${LOOPS_TRIGGERS_API_PATH}/${triggerId}`,
      payload,
    );
    return data;
  },

  fireTrigger: async (
    triggerId: string,
    context?: Record<string, unknown>,
  ): Promise<FireTriggerResult> => {
    const { data } = await loopAxios.post<FireTriggerResult>(
      `${LOOPS_TRIGGERS_API_PATH}/${triggerId}/fire`,
      context ?? {},
    );
    return data;
  },

  listTriggerEvents: async (triggerId: string): Promise<TriggerEvent[]> => {
    const { data } = await loopAxios.get<TriggerEvent[]>(
      `${LOOPS_TRIGGERS_API_PATH}/${triggerId}/events`,
    );
    return data;
  },

  listEvents: async (): Promise<TriggerEvent[]> => {
    const { data } = await loopAxios.get<TriggerEvent[]>(
      `${LOOPS_TRIGGERS_API_PATH}/events`,
    );
    return data;
  },
};

export default LoopService;
