import React from "react";
import { render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { WebSocketProviderWrapper } from "#/contexts/websocket-provider-wrapper";

const state = vi.hoisted(() => ({
  isResuming: true,
  sandboxStatus: "RUNNING",
  providerProps: vi.fn(),
  recoverCredentialBinding: vi.fn(),
}));

vi.mock("#/contexts/conversation-websocket-context", () => ({
  ConversationWebSocketProvider: (props: Record<string, unknown>) => {
    state.providerProps(props);
    return props.children;
  },
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: {
      sandbox_status: state.sandboxStatus,
      conversation_url: "http://agent-server/api/conversations/conversation-id",
      session_api_key: "session-key",
      sub_conversation_ids: [],
    },
    refetch: vi.fn(),
    isFetched: true,
  }),
}));

vi.mock("#/hooks/query/use-sub-conversations", () => ({
  useSubConversations: () => ({ data: [] }),
}));

vi.mock("#/hooks/use-sandbox-recovery", () => ({
  useSandboxRecovery: () => ({
    isResuming: state.isResuming,
    recoverCredentialBinding: state.recoverCredentialBinding,
  }),
}));

describe("WebSocketProviderWrapper", () => {
  beforeEach(() => {
    state.isResuming = true;
    state.sandboxStatus = "RUNNING";
    state.providerProps.mockClear();
    state.recoverCredentialBinding.mockClear();
  });

  it("withholds runtime credentials until resume completes", () => {
    const { rerender } = render(
      <WebSocketProviderWrapper conversationId="conversation-id">
        child
      </WebSocketProviderWrapper>,
    );

    expect(state.providerProps.mock.lastCall?.[0]).toMatchObject({
      conversationUrl: undefined,
      sessionApiKey: undefined,
    });

    state.isResuming = false;
    rerender(
      <WebSocketProviderWrapper conversationId="conversation-id">
        child
      </WebSocketProviderWrapper>,
    );

    expect(state.providerProps.mock.lastCall?.[0]).toMatchObject({
      conversationUrl: "http://agent-server/api/conversations/conversation-id",
      sessionApiKey: "session-key",
      onCredentialBindingActivationRequired: state.recoverCredentialBinding,
    });
  });
});
