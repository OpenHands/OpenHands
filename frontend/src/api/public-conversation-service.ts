import { OpenHandsAxios } from "./open-hands-axios";

export interface PublicSharingRequest {
  is_public: boolean;
}

export interface PublicSharingResponse {
  is_public: boolean;
  share_token?: string;
  share_url?: string;
}

export interface PublicConversationInfo {
  conversation_id: string;
  title: string;
  status: string;
  selected_repository?: string;
  selected_branch?: string;
  git_provider?: string;
  trigger?: string;
  created_at: string;
  last_updated_at?: string;
  shared_at?: string;
}

export interface PublicMessageInfo {
  id: string;
  timestamp: string;
  source: string;
  content: string;
}

export interface PublicConversationDetail {
  conversation: PublicConversationInfo;
  messages: PublicMessageInfo[];
}

export class PublicConversationService {
  /**
   * Update the public sharing status of a conversation
   */
  static async updatePublicSharing(
    conversationId: string,
    data: PublicSharingRequest,
  ): Promise<PublicSharingResponse> {
    const response = await OpenHandsAxios.post(
      `/api/conversations/${conversationId}/public-sharing`,
      data,
    );
    return response.data;
  }

  /**
   * Get the current public sharing status of a conversation
   */
  static async getPublicSharing(
    conversationId: string,
  ): Promise<PublicSharingResponse> {
    const response = await OpenHandsAxios.get(
      `/api/conversations/${conversationId}/public-sharing`,
    );
    return response.data;
  }

  /**
   * Get public conversation info (no authentication required)
   */
  static async getPublicConversation(
    conversationId: string,
  ): Promise<PublicConversationInfo> {
    const response = await OpenHandsAxios.get(
      `/api/public/conversations/${conversationId}`,
    );
    return response.data;
  }

  /**
   * Get public conversation messages (no authentication required)
   */
  static async getPublicConversationMessages(
    conversationId: string,
  ): Promise<PublicMessageInfo[]> {
    const response = await OpenHandsAxios.get(
      `/api/public/conversations/${conversationId}/messages`,
    );
    return response.data;
  }

  /**
   * Get complete public conversation with messages (no authentication required)
   */
  static async getPublicConversationFull(
    conversationId: string,
  ): Promise<PublicConversationDetail> {
    const response = await OpenHandsAxios.get(
      `/api/public/conversations/${conversationId}/full`,
    );
    return response.data;
  }

  /**
   * Get public conversation by share token (no authentication required)
   */
  static async getPublicConversationByToken(
    shareToken: string,
  ): Promise<PublicConversationInfo> {
    const response = await OpenHandsAxios.get(
      `/api/public/conversations/token/${shareToken}`,
    );
    return response.data;
  }
}
