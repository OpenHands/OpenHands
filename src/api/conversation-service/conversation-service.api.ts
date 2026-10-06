import { RemoteEventsList } from "@openhands/typescript-client/events/remote-events-list";
import { uploadFilesToConversation } from "#/api/conversation-file-upload.api";
import {
  GetTrajectoryResponse,
  FileUploadSuccessResponse,
} from "../open-hands.types";
import { getAgentServerHttpClientOptions } from "../agent-server-client-options";
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

  static async getTrajectory(
    conversationId: string,
  ): Promise<GetTrajectoryResponse> {
    const page = await new RemoteEventsList(
      getAgentServerHttpClientOptions(this.getClientOverrides(conversationId)),
      conversationId,
    ).search({ limit: 10000 });

    return { trajectory: page.items ?? [] };
  }

  static async uploadFiles(
    conversationId: string,
    files: File[],
  ): Promise<FileUploadSuccessResponse> {
    return uploadFilesToConversation(
      conversationId,
      files,
      this.currentConversation,
    );
  }
}

export default ConversationService;
