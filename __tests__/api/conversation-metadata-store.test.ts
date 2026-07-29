import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { toAppConversation } from "#/api/agent-server-adapter";
import {
  getStoredConversationMetadata,
  removeStoredConversationMetadata,
  setStoredConversationMetadata,
} from "#/api/conversation-metadata-store";

const directInfo = (id: string) => ({
  id,
  created_at: "2026-05-05T00:00:00Z",
  updated_at: "2026-05-05T00:00:00Z",
});

describe("conversation-metadata-store", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });
  afterEach(() => {
    window.localStorage.clear();
  });

  it("toAppConversation hydrates repo/branch/git_provider from the store", () => {
    setStoredConversationMetadata("conv-1", {
      selected_repository: "octocat/hello-world",
      selected_branch: "main",
      git_provider: "github",
    });

    const conversation = toAppConversation(directInfo("conv-1"));

    expect(conversation).toMatchObject({
      selected_repository: "octocat/hello-world",
      selected_branch: "main",
      git_provider: "github",
    });
  });

  it("toAppConversation falls back to nulls when no metadata is stored", () => {
    const conversation = toAppConversation(directInfo("conv-2"));

    expect(conversation.selected_repository).toBeNull();
    expect(conversation.selected_branch).toBeNull();
    expect(conversation.git_provider).toBeNull();
  });

  it("removeStoredConversationMetadata clears the entry so the adapter goes back to nulls", () => {
    setStoredConversationMetadata("conv-3", {
      selected_repository: "octocat/repo",
      selected_branch: "trunk",
      git_provider: "github",
    });

    removeStoredConversationMetadata("conv-3");

    expect(getStoredConversationMetadata("conv-3")).toBeNull();
    expect(toAppConversation(directInfo("conv-3")).selected_repository).toBeNull();
  });
});
