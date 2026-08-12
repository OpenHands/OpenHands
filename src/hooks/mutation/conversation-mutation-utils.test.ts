import { describe, expect, it } from "vitest";
import { QueryClient } from "@tanstack/react-query";
import {
  resolveLaunchProfile,
  updateConversationExecutionStatusInCache,
  updateConversationLlmModelInCache,
} from "./conversation-mutation-utils";
import type { AgentProfileListResponse } from "#/api/agent-profiles-service/agent-profiles-service.api";
import type { ProfileListResponse } from "#/api/profiles-service/profiles-service.api";
import { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";

const agentProfiles = (profile: Record<string, unknown>) =>
  ({
    profiles: [profile],
    active_agent_profile_id: profile.id,
  }) as unknown as AgentProfileListResponse;

const llmProfiles = (activeProfile: string, ...names: string[]) =>
  ({
    active_profile: activeProfile,
    profiles: names.map((name) => ({ name })),
  }) as unknown as ProfileListResponse;

const createConversation = (): AppConversation => ({
  id: "conversation-1",
  created_by_user_id: null,
  selected_repository: null,
  selected_branch: null,
  git_provider: null,
  title: "Test conversation",
  trigger: null,
  pr_number: [],
  llm_model: null,
  metrics: null,
  created_at: "2026-04-16T00:00:00Z",
  updated_at: "2026-04-16T00:00:00Z",
  execution_status: ExecutionStatus.RUNNING,
  conversation_url: "http://localhost:3000/api/conversations/conversation-1",
  session_api_key: "session-key",
  sandbox_id: null,
  sub_conversation_ids: [],
});

describe("resolveLaunchProfile", () => {
  it("honors a home dropdown selection over a named profile's pinned model", () => {
    const profile = {
      id: "agent-profile-1",
      name: "openhands-luna",
      agent_kind: "openhands",
      llm_profile_ref: "pinned-model",
    };

    expect(
      resolveLaunchProfile({
        requestedAgentProfileId: profile.id as string,
        agentProfiles: agentProfiles(profile),
        llmProfiles: llmProfiles(
          "selected-model",
          "pinned-model",
          "selected-model",
        ),
        isCloud: false,
      }),
    ).toMatchObject({
      effectiveAgentProfileId: undefined,
      launchLlmProfileRef: "selected-model",
      downgradeReason: "dropdown-llm-profile-selected",
    });
  });

  it("keeps the named profile path when its pinned model is selected", () => {
    const profile = {
      id: "agent-profile-1",
      name: "openhands-luna",
      agent_kind: "openhands",
      llm_profile_ref: "pinned-model",
    };

    expect(
      resolveLaunchProfile({
        requestedAgentProfileId: profile.id as string,
        agentProfiles: agentProfiles(profile),
        llmProfiles: llmProfiles("pinned-model", "pinned-model"),
        isCloud: false,
      }),
    ).toMatchObject({
      effectiveAgentProfileId: profile.id,
      launchLlmProfileRef: "pinned-model",
    });
  });

  it("uses the active LLM profile when no agent profile is requested", () => {
    expect(
      resolveLaunchProfile({
        llmProfiles: llmProfiles("selected-model", "selected-model"),
        isCloud: false,
      }),
    ).toEqual({ launchLlmProfileRef: "selected-model" });
  });
});

describe("updateConversationExecutionStatusInCache", () => {
  it("updates the active conversation execution_status field", () => {
    const queryClient = new QueryClient();
    const conversation = createConversation();

    queryClient.setQueryData(
      ["user", "conversation", conversation.id],
      conversation,
    );

    updateConversationExecutionStatusInCache(
      queryClient,
      conversation.id,
      ExecutionStatus.PAUSED,
    );

    expect(
      queryClient.getQueryData<AppConversation | null>([
        "user",
        "conversation",
        conversation.id,
      ]),
    ).toMatchObject({
      execution_status: ExecutionStatus.PAUSED,
    });
  });
});

describe("updateConversationLlmModelInCache", () => {
  it("updates active conversation and list cache entries", () => {
    const queryClient = new QueryClient();
    const conversation = createConversation();
    const otherConversation = { ...createConversation(), id: "conversation-2" };

    queryClient.setQueryData(
      ["user", "conversation", conversation.id, "backend-1", null],
      conversation,
    );
    queryClient.setQueryData(["user", "conversations"], {
      pages: [
        {
          items: [conversation, otherConversation],
        },
      ],
    });

    updateConversationLlmModelInCache(
      queryClient,
      conversation.id,
      "anthropic/claude-haiku-4-5",
    );

    expect(
      queryClient.getQueryData<AppConversation | null>([
        "user",
        "conversation",
        conversation.id,
        "backend-1",
        null,
      ]),
    ).toMatchObject({
      llm_model: "anthropic/claude-haiku-4-5",
    });

    expect(
      queryClient.getQueryData<{
        pages: Array<{ items: AppConversation[] }>;
      }>(["user", "conversations"])?.pages[0].items,
    ).toEqual([
      expect.objectContaining({
        id: conversation.id,
        llm_model: "anthropic/claude-haiku-4-5",
      }),
      expect.objectContaining({
        id: otherConversation.id,
        llm_model: null,
      }),
    ]);
  });
});
