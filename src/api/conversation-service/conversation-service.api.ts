import { VSCodeClient } from "@openhands/typescript-client/clients";
import { RemoteEventsList } from "@openhands/typescript-client/events/remote-events-list";
import { uploadFilesToConversation } from "#/api/conversation-file-upload.api";
import {
  GetVSCodeUrlResponse,
  GetTrajectoryResponse,
  FileUploadSuccessResponse,
} from "../open-hands.types";
import { getAgentServerWorkingDir } from "../agent-server-config";
import {
  getAgentServerClientOptions,
  getAgentServerHttpClientOptions,
} from "../agent-server-client-options";
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

  private static getClientOverrides() {
    return {
      sessionApiKey: this.currentConversation?.session_api_key,
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
      getAgentServerClientOptions(this.getClientOverrides()),
    ).getUrl({
      baseUrl:
        typeof window !== "undefined" ? window.location.origin : undefined,
      workspaceDir,
    });

    return { vscode_url: vscodeUrl };
  }

  static async getTrajectory(
    conversationId: string,
  ): Promise<GetTrajectoryResponse> {
    const page = await new RemoteEventsList(
      getAgentServerHttpClientOptions(this.getClientOverrides()),
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
