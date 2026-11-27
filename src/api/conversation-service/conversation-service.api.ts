import { VSCodeClient } from "@openhands/typescript-client/clients";
import { GetVSCodeUrlResponse } from "../open-hands.types";
import { getAgentServerWorkingDir } from "../agent-server-config";
import { getAgentServerClientOptions } from "../agent-server-client-options";
import { AppConversation } from "./agent-server-conversation-service.types";

class ConversationService {
  private static currentConversation: AppConversation | null = null;

  static setCurrentConversation(
    currentConversation: AppConversation | null,
  ): void {
    this.currentConversation = currentConversation;
  }

  static getCurrentConversation(): AppConversation | null {
    return this.currentConversation;
  }

  private static getClientOverrides(conversationId: string) {
    return {
      conversationId,
      ...(this.currentConversation?.id === conversationId
        ? {
            conversationUrl: this.currentConversation.conversation_url,
            sessionApiKey: this.currentConversation.session_api_key,
          }
        : {}),
    };
  }

  static async getVSCodeUrl(
    conversationId: string,
  ): Promise<GetVSCodeUrlResponse> {
    const workspaceDir =
      this.currentConversation?.id === conversationId
        ? (this.currentConversation?.workspace?.working_dir ??
          getAgentServerWorkingDir())
        : getAgentServerWorkingDir();
    const vscodeUrl = await new VSCodeClient(
      getAgentServerClientOptions(this.getClientOverrides(conversationId)),
    ).getUrl({
      baseUrl:
        typeof window !== "undefined" ? window.location.origin : undefined,
      workspaceDir,
    });

    return { vscode_url: vscodeUrl };
  }
}

export default ConversationService;
