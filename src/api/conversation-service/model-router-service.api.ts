import {
  ConversationClient,
  type ConversationClientOptions,
} from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";

export interface ModelRouterRouteInfo {
  name: string;
  model?: string | null;
  label?: string | null;
}

export interface ModelRouterSelection {
  mode: "auto" | "manual";
  manual_route_name?: string | null;
}

export interface ModelRouterStatus {
  enabled: boolean;
  selection?: ModelRouterSelection | null;
  last_route_name?: string | null;
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
    profileName: string,
    routeName?: string,
    baseRouteName = "default",
  ): Promise<ModelRouterStatus> {
    const response = await this.transport.post<ModelRouterStatus>(
      `/api/conversations/${conversationId}/model_router/routes`,
      {
        profile_name: profileName,
        route_name: routeName,
        base_route_name: baseRouteName,
      },
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
    routeName: string,
  ): Promise<ModelRouterStatus> {
    return this.updateModelRouter(conversationId, {
      mode: "manual",
      manual_route_name: routeName,
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
    routeName: string,
  ): Promise<ModelRouterStatus> {
    return modelRouterClient(
      conversationUrl,
      sessionApiKey,
    ).addModelRouterRoute(conversationId, profileName, routeName);
  }

  static async switchRoute(
    conversationId: string,
    conversationUrl: string | null | undefined,
    sessionApiKey: string | null | undefined,
    routeName: string,
  ): Promise<ModelRouterStatus> {
    return modelRouterClient(conversationUrl, sessionApiKey).switchModelRouter(
      conversationId,
      routeName,
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
