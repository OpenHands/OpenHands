import { beforeEach, describe, expect, it, vi } from "vitest";
import { ConversationClient } from "@openhands/typescript-client/clients";
import ConversationService from "./conversation-service.api";
import AgentServerConversationService from "./agent-server-conversation-service.api";
import {
  syncSecretToActiveConversation,
  removeSecretFromActiveConversation,
  renameSecretInActiveConversation,
} from "./conversation-secret-sync";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "../backend-registry/active-store";
import { callCloudProxy } from "../cloud/proxy";
import type { AppConversation } from "./agent-server-conversation-service.types";

vi.mock("@openhands/typescript-client/clients", () => ({
  ConversationClient: vi.fn(),
  FileClient: vi.fn(),
  ProfilesClient: vi.fn(),
  VSCodeClient: vi.fn(),
}));

vi.mock("../cloud/proxy", () => ({
  callCloudProxy: vi.fn(),
}));

const updateSecretsMock = vi.fn();

function useBackend(kind: "local" | "cloud"): void {
  setRegisteredBackends([
    {
      id: kind,
      name: kind,
      host: "http://127.0.0.1:8001",
      apiKey: "session-key",
      kind,
    },
  ]);
  setActiveSelection({ backendId: kind, orgId: null });
}

describe("AgentServerConversationService.updateSecrets", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(ConversationClient).mockImplementation(
      function MockConversationClient() {
        return {
          updateSecrets: updateSecretsMock,
        } as unknown as ConversationClient;
      } as unknown as typeof ConversationClient,
    );
  });

  it("calls ConversationClient.updateSecrets on local backend", async () => {
    useBackend("local");
    const secrets = {
      API_KEY: {
        kind: "LookupSecret" as const,
        url: "/api/settings/secrets/API_KEY",
        description: "Test API Key",
      },
    };

    await AgentServerConversationService.updateSecrets("convo-123", secrets);

    expect(updateSecretsMock).toHaveBeenCalledWith("convo-123", { secrets });
    expect(callCloudProxy).not.toHaveBeenCalled();
  });

  it("calls callCloudProxy on cloud backend", async () => {
    useBackend("cloud");
    const secrets = {
      API_KEY: {
        kind: "LookupSecret" as const,
        url: "/api/settings/secrets/API_KEY",
        description: "Test API Key",
      },
    };

    await AgentServerConversationService.updateSecrets("convo-123", secrets);

    expect(callCloudProxy).toHaveBeenCalledWith({
      backend: expect.objectContaining({ kind: "cloud" }),
      method: "POST",
      path: "/api/v1/app-conversations/convo-123/secrets",
      body: { secrets },
    });
    expect(ConversationClient).not.toHaveBeenCalled();
  });
});

describe("conversation-secret-sync helpers", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useBackend("local");
    ConversationService.setCurrentConversation(null);
    vi.mocked(ConversationClient).mockImplementation(
      function MockConversationClient() {
        return {
          updateSecrets: updateSecretsMock,
        } as unknown as ConversationClient;
      } as unknown as typeof ConversationClient,
    );
  });

  it("does nothing if there is no active conversation", async () => {
    ConversationService.setCurrentConversation(null);

    await syncSecretToActiveConversation("NEW_KEY", "Description");

    expect(updateSecretsMock).not.toHaveBeenCalled();
  });

  it("syncs newly created or updated secret as a LookupSecret to active conversation", async () => {
    const mockConvo = { id: "active-convo-1" } as AppConversation;
    ConversationService.setCurrentConversation(mockConvo);

    await syncSecretToActiveConversation("MY_SECRET", "My Secret Desc");

    expect(updateSecretsMock).toHaveBeenCalledWith("active-convo-1", {
      secrets: {
        MY_SECRET: expect.objectContaining({
          kind: "LookupSecret",
          url: "/api/settings/secrets/MY_SECRET",
          description: "My Secret Desc",
        }),
      },
    });
  });

  it("removes a secret from active conversation using a null StaticSecret placeholder", async () => {
    const mockConvo = { id: "active-convo-1" } as AppConversation;
    ConversationService.setCurrentConversation(mockConvo);

    await removeSecretFromActiveConversation("OLD_SECRET");

    expect(updateSecretsMock).toHaveBeenCalledWith("active-convo-1", {
      secrets: {
        OLD_SECRET: {
          kind: "StaticSecret",
          value: null,
        },
      },
    });
  });

  it("renames a secret in active conversation by clearing old key and adding new key", async () => {
    const mockConvo = { id: "active-convo-1" } as AppConversation;
    ConversationService.setCurrentConversation(mockConvo);

    await renameSecretInActiveConversation(
      "OLD_KEY",
      "NEW_KEY",
      "New description",
    );

    expect(updateSecretsMock).toHaveBeenNthCalledWith(1, "active-convo-1", {
      secrets: {
        OLD_KEY: {
          kind: "StaticSecret",
          value: null,
        },
      },
    });
    expect(updateSecretsMock).toHaveBeenNthCalledWith(2, "active-convo-1", {
      secrets: {
        NEW_KEY: expect.objectContaining({
          kind: "LookupSecret",
          url: "/api/settings/secrets/NEW_KEY",
          description: "New description",
        }),
      },
    });
  });
});
