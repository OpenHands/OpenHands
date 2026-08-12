import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { useConversationHooks } from "#/hooks/query/use-conversation-hooks";

function HooksQueryProbe() {
  const query = useConversationHooks(null);
  return <span>{query.fetchStatus}</span>;
}

describe("useConversationHooks", () => {
  // @spec SC-003 — Loaded extensions
  it("stays idle when the chat input has no active conversation", () => {
    const getHooks = vi.spyOn(AgentServerConversationService, "getHooks");

    renderWithProviders(<HooksQueryProbe />, {
      navigation: { conversationId: null },
    });

    expect(screen.getByText("idle")).toBeInTheDocument();
    expect(getHooks).not.toHaveBeenCalled();
  });
});
