import { openHands } from "#/api/open-hands-axios";

export type ChatgptDeviceSessionResponse = {
  session_id: string;
  user_code: string;
  verification_uri: string;
};

export type ChatgptPollResponse = {
  status: "pending" | "complete";
};

export const ChatgptOauthApi = {
  startDeviceSession: async (): Promise<ChatgptDeviceSessionResponse> => {
    const { data } = await openHands.post<ChatgptDeviceSessionResponse>(
      "/api/v1/llm/chatgpt/device-session",
    );
    return data;
  },

  pollSession: async (sessionId: string): Promise<ChatgptPollResponse> => {
    const { data } = await openHands.get<ChatgptPollResponse>(
      `/api/v1/llm/chatgpt/poll/${sessionId}`,
    );
    return data;
  },
};
