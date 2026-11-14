import { v4 as uuidv4 } from "uuid";
import type { LLMConfig } from "@openhands/typescript-client";
import {
  ConversationClient,
  type ConversationClientOptions,
} from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import { assertSubscriptionAuthReady } from "#/api/agent-server-adapter";

export interface ModelRouterRouteInfo {
  key: string;
  model: string;
  label: string;
}

export interface ModelRouterSelection {
  mode: "auto" | "manual";
  manual_route_key?: string | null;
}

export interface ModelRouterStatus {
  enabled: boolean;
  selection: ModelRouterSelection | null;
  last_route_key: string | null;
  routes: ModelRouterRouteInfo[];
}

interface ConversationClientTransport {
  get<T>(path: string): Promise<{ data: T }>;
  post<T>(path: string, body?: unknown): Promise<{ data: T }>;
  patch<T>(path: string, body?: unknown): Promise<{ data: T }>;
}

interface ConversationClientTransportCarrier {
  client: ConversationClientTransport;
}

class ModelRouterConversationClient extends ConversationClient {
  private get transport(): ConversationClientTransport {
    return (this as unknown as ConversationClientTransportCarrier).client;
  }

  async getModelRouter(conversationId: string): Promise<ModelRouterStatus> {
    const response = await this.transport.get<ModelRouterStatus>(
      `/api/conversations/${conversationId}/model_router`,
    );
    return response.data;
  }

  async addModelRouterRoute(
    conversationId: string,
    routeKey: string,
    llm: LLMConfig,
    baseRouteKey = "default",
  ): Promise<ModelRouterStatus> {
    const response = await this.transport.post<ModelRouterStatus>(
      `/api/conversations/${conversationId}/model_router/routes`,
      { route_key: routeKey, llm, base_route_key: baseRouteKey },
    );
    return response.data;
  }

  async updateModelRouter(
    conversationId: string,
    selection: ModelRouterSelection,
  ): Promise<ModelRouterStatus> {
    const response = await this.transport.patch<ModelRouterStatus>(
      `/api/conversations/${conversationId}/model_router`,
      { selection },
    );
    return response.data;
  }

  async switchModelRouter(
    conversationId: string,
    routeKey: string,
  ): Promise<ModelRouterStatus> {
    return this.updateModelRouter(conversationId, {
      mode: "manual",
      manual_route_key: routeKey,
    });
  }

  async resumeAutoModelRouting(
    conversationId: string,
  ): Promise<ModelRouterStatus> {
    return this.updateModelRouter(conversationId, { mode: "auto" });
  }
}

function clientOptionsForConversation(
  conversationUrl: string | null | undefined,
  sessionApiKey?: string | null,
): ConversationClientOptions {
  return getAgentServerClientOptions({ conversationUrl, sessionApiKey });
}

function modelRouterClient(
  conversationUrl: string | null | undefined,
  sessionApiKey?: string | null,
): ModelRouterConversationClient {
  return new ModelRouterConversationClient(
    clientOptionsForConversation(conversationUrl, sessionApiKey),
  );
}

class ModelRouterService {
  static async getModelRouter(
    conversationId: string,
    conversationUrl: string | null | undefined,
    sessionApiKey?: string | null,
  ): Promise<ModelRouterStatus> {
    return modelRouterClient(conversationUrl, sessionApiKey).getModelRouter(
      conversationId,
    );
  }

  static async addProfileRoute(
    conversationId: string,
    conversationUrl: string | null | undefined,
    sessionApiKey: string | null | undefined,
    profileName: string,
    routeKey: string,
  ): Promise<ModelRouterStatus> {
    const profile = await ProfilesService.getProfile(profileName, "encrypted");
    const model =
      typeof profile.config.model === "string" ? profile.config.model : "";
    if (!model) throw new Error(`Profile '${profileName}' has no model.`);

    await assertSubscriptionAuthReady({ llm: profile.config });

    return modelRouterClient(
      conversationUrl,
      sessionApiKey,
    ).addModelRouterRoute(conversationId, routeKey, {
      ...profile.config,
      model,
      stream: true,
      usage_id: `profile:${profileName}:${uuidv4()}`,
    } as LLMConfig);
  }

  static async switchRoute(
    conversationId: string,
    conversationUrl: string | null | undefined,
    sessionApiKey: string | null | undefined,
    routeKey: string,
  ): Promise<ModelRouterStatus> {
    return modelRouterClient(conversationUrl, sessionApiKey).switchModelRouter(
      conversationId,
      routeKey,
    );
  }

  static async resumeAutoRouting(
    conversationId: string,
    conversationUrl: string | null | undefined,
    sessionApiKey?: string | null,
  ): Promise<ModelRouterStatus> {
    return modelRouterClient(
      conversationUrl,
      sessionApiKey,
    ).resumeAutoModelRouting(conversationId);
  }
}

export default ModelRouterService;
