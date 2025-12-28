import { AxiosHeaders } from "axios";
import {
  Feedback,
  FeedbackResponse,
  GetVSCodeUrlResponse,
  Conversation,
  ResultSet,
  GetTrajectoryResponse,
  GetMicroagentsResponse,
  GetMicroagentPromptResponse,
  CreateMicroagent,
  FileUploadSuccessResponse,
  GetFilesResponse,
} from "../open-hands.types";
import { openHands } from "../open-hands-axios";
import { Provider } from "#/types/settings";
import { SuggestedTask } from "#/utils/types";
import { BatchFeedbackData } from "#/hooks/query/use-batch-feedback";

class ConversationService {
  private static currentConversation: Conversation | null = null;

  /**
   * Get a current conversation
   * @return the current conversation
   */
  static getCurrentConversation(): Conversation | null {
    return this.currentConversation;
  }

  /**
   * Set a current conversation
   * @param url Custom URL to use for conversation endpoints
   */
  static setCurrentConversation(
    currentConversation: Conversation | null,
  ): void {
    this.currentConversation = currentConversation;
  }

  /**
   * Get the url for the conversation. If
   */
  static getConversationUrl(conversationId: string): string {
    if (this.currentConversation?.conversation_id === conversationId) {
      if (this.currentConversation.url) {
        return this.currentConversation.url;
      }
    }
    return `/api/conversations/${conversationId}`;
  }

  static getConversationHeaders(): AxiosHeaders {
    const headers = new AxiosHeaders();
    const sessionApiKey = this.currentConversation?.session_api_key;
    if (sessionApiKey) {
      headers.set("X-Session-API-Key", sessionApiKey);
    }
    return headers;
  }

  /**
   * Send feedback to the server
   * @param data Feedback data
   * @returns The stored feedback data
   */
  static async submitFeedback(
    conversationId: string,
    feedback: Feedback,
  ): Promise<FeedbackResponse> {
    const url = `/api/conversations/${conversationId}/submit-feedback`;
    const { data } = await openHands.post<FeedbackResponse>(url, feedback);
    return data;
  }

  /**
   * Submit conversation feedback with rating
   * @param conversationId The conversation ID
   * @param rating The rating (1-5)
   * @param eventId Optional event ID this feedback corresponds to
   * @param reason Optional reason for the rating
   * @returns Response from the feedback endpoint
   */
  static async submitConversationFeedback(
    conversationId: string,
    rating: number,
    eventId?: number,
    reason?: string,
  ): Promise<{ status: string; message: string }> {
    const url = `/feedback/conversation`;
    const payload = {
      conversation_id: conversationId,
      event_id: eventId,
      rating,
      reason,
      metadata: { source: "likert-scale" },
    };
    const { data } = await openHands.post<{ status: string; message: string }>(
      url,
      payload,
    );
    return data;
  }

  /**
   * Check if feedback exists for a specific conversation and event
   * @param conversationId The conversation ID
   * @param eventId The event ID to check
   * @returns Feedback data including existence, rating, and reason
   */
  static async checkFeedbackExists(
    conversationId: string,
    eventId: number,
  ): Promise<{ exists: boolean; rating?: number; reason?: string }> {
    try {
      const url = `/feedback/conversation/${conversationId}/${eventId}`;
      const { data } = await openHands.get<{
        exists: boolean;
        rating?: number;
        reason?: string;
      }>(url);
      return data;
    } catch {
      // Error checking if feedback exists
      return { exists: false };
    }
  }

  /**
   * Get feedback for multiple events in a conversation
   * @param conversationId The conversation ID
   * @returns Map of event IDs to feedback data including existence, rating, reason and metadata
   */
  static async getBatchFeedback(conversationId: string): Promise<
    Record<
      string,
      {
        exists: boolean;
        rating?: number;
        reason?: string;
        metadata?: Record<string, BatchFeedbackData>;
      }
    >
  > {
    const url = `/feedback/conversation/${conversationId}/batch`;
    const { data } = await openHands.get<
      Record<
        string,
        {
          exists: boolean;
          rating?: number;
          reason?: string;
          metadata?: Record<string, BatchFeedbackData>;
        }
      >
    >(url);

    return data;
  }

  /**
   * Get the web hosts
   * @returns Array of web hosts
   */
  static async getWebHosts(conversationId: string): Promise<string[]> {
    const url = `${this.getConversationUrl(conversationId)}/web-hosts`;
    const response = await openHands.get(url, {
      headers: this.getConversationHeaders(),
    });
    return Object.keys(response.data.hosts);
  }

  /**
   * Get the VSCode URL
   * @returns VSCode URL
   */
  static async getVSCodeUrl(
    conversationId: string,
  ): Promise<GetVSCodeUrlResponse> {
    const url = `${this.getConversationUrl(conversationId)}/vscode-url`;
    const { data } = await openHands.get<GetVSCodeUrlResponse>(url, {
      headers: this.getConversationHeaders(),
    });
    return data;
  }

  static async getRuntimeId(
    conversationId: string,
  ): Promise<{ runtime_id: string }> {
    const url = `${this.getConversationUrl(conversationId)}/config`;
    const { data } = await openHands.get<{ runtime_id: string }>(url, {
      headers: this.getConversationHeaders(),
    });
    return data;
  }

  static async getUserConversations(
    limit: number = 20,
    pageId?: string,
  ): Promise<ResultSet<Conversation>> {
    const params = new URLSearchParams();
    params.append("limit", limit.toString());

    if (pageId) {
      params.append("page_id", pageId);
    }

    const { data } = await openHands.get<ResultSet<Conversation>>(
      `/api/conversations?${params.toString()}`,
    );
    return data;
  }

  static async searchConversations(
    selectedRepository?: string,
    conversationTrigger?: string,
    limit: number = 100,
  ): Promise<Conversation[]> {
    const params = new URLSearchParams();
    params.append("limit", limit.toString());

    if (selectedRepository) {
      params.append("selected_repository", selectedRepository);
    }

    if (conversationTrigger) {
      params.append("conversation_trigger", conversationTrigger);
    }

    const { data } = await openHands.get<ResultSet<Conversation>>(
      `/api/conversations?${params.toString()}`,
    );
    return data.results;
  }

  static async deleteUserConversation(conversationId: string): Promise<void> {
    await openHands.delete(`/api/conversations/${conversationId}`);
  }

  static async createConversation(
    selectedRepository?: string,
    git_provider?: Provider,
    initialUserMsg?: string,
    suggested_task?: SuggestedTask,
    selected_branch?: string,
    conversationInstructions?: string,
    createMicroagent?: CreateMicroagent,
  ): Promise<Conversation> {
    const body = {
      repository: selectedRepository,
      git_provider,
      selected_branch,
      initial_user_msg: initialUserMsg,
      suggested_task,
      conversation_instructions: conversationInstructions,
      create_microagent: createMicroagent,
    };

    const { data } = await openHands.post<Conversation>(
      "/api/conversations",
      body,
    );

    return data;
  }

  static async getConversation(
    conversationId: string,
  ): Promise<Conversation | null> {
    const { data } = await openHands.get<Conversation | null>(
      `/api/conversations/${conversationId}`,
    );

    return data;
  }

  static async startConversation(
    conversationId: string,
    providers?: Provider[],
  ): Promise<Conversation | null> {
    const { data } = await openHands.post<Conversation | null>(
      `/api/conversations/${conversationId}/start`,
      providers ? { providers_set: providers } : {},
    );

    return data;
  }

  static async stopConversation(
    conversationId: string,
  ): Promise<Conversation | null> {
    const { data } = await openHands.post<Conversation | null>(
      `/api/conversations/${conversationId}/stop`,
    );

    return data;
  }

  static async getTrajectory(
    conversationId: string,
  ): Promise<GetTrajectoryResponse> {
    const url = `${this.getConversationUrl(conversationId)}/trajectory`;
    const { data } = await openHands.get<GetTrajectoryResponse>(url, {
      headers: this.getConversationHeaders(),
    });
    return data;
  }

  /**
   * Get the available microagents associated with a conversation
   * @param conversationId The ID of the conversation
   * @returns The available microagents associated with the conversation
   */
  static async getMicroagents(
    conversationId: string,
  ): Promise<GetMicroagentsResponse> {
    const url = `${this.getConversationUrl(conversationId)}/microagents`;
    const { data } = await openHands.get<GetMicroagentsResponse>(url, {
      headers: this.getConversationHeaders(),
    });
    return data;
  }

  static async getMicroagentPrompt(
    conversationId: string,
    eventId: number,
  ): Promise<string> {
    const url = `${this.getConversationUrl(conversationId)}/remember-prompt`;
    const { data } = await openHands.get<GetMicroagentPromptResponse>(url, {
      params: { event_id: eventId },
      headers: this.getConversationHeaders(),
    });

    return data.prompt;
  }

  static async updateConversation(
    conversationId: string,
    updates: { title: string },
  ): Promise<boolean> {
    const { data } = await openHands.patch<boolean>(
      `/api/conversations/${conversationId}`,
      updates,
    );

    return data;
  }

  /**
   * Retrieve the list of files available in the workspace
   * @param conversationId ID of the conversation
   * @param path Path to list files from. If provided, it lists all the files in the given path
   * @returns List of files available in the given path. If path is not provided, it lists all the files in the workspace
   */
  static async getFiles(
    conversationId: string,
    path?: string,
  ): Promise<GetFilesResponse> {
    const url = `${this.getConversationUrl(conversationId)}/list-files`;
    const { data } = await openHands.get<GetFilesResponse>(url, {
      params: { path },
      headers: this.getConversationHeaders(),
    });

    return data;
  }

  /**
   * Upload multiple files to the workspace
   * @param conversationId ID of the conversation
   * @param files List of files.
   * @returns list of uploaded files, list of skipped files
   */
  static async uploadFiles(
    conversationId: string,
    files: File[],
  ): Promise<FileUploadSuccessResponse> {
    const formData = new FormData();
    for (const file of files) {
      formData.append("files", file);
    }
    const url = `${this.getConversationUrl(conversationId)}/upload-files`;
    const response = await openHands.post<FileUploadSuccessResponse>(
      url,
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
          ...this.getConversationHeaders(),
        },
      },
    );
    return response.data;
  }
}

export default ConversationService;
