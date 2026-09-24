import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { SharedConversation as SharedConversationData } from "@openhands/typescript-client";

import SharedConversation from "#/routes/shared-conversation";
import { I18nKey } from "#/i18n/declaration";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  getCloudSharedConversation,
  searchCloudSharedEvents,
} from "#/api/cloud/shared-conversation-service.api";
import { getCloudOrganizationMember } from "#/api/cloud/organization-service.api";

// Mock the cloud services the shared-conversation queries depend on.
vi.mock("#/api/cloud/shared-conversation-service.api", () => ({
  getCloudSharedConversation: vi.fn(),
  searchCloudSharedEvents: vi.fn(),
}));
vi.mock("#/api/cloud/organization-service.api", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("#/api/cloud/organization-service.api")
  >()),
  getCloudOrganizationMember: vi.fn(),
}));

// The message list is exercised by its own tests; keep this one on the header.
vi.mock("#/components/conversation-events/chat/messages", () => ({
  Messages: () => <section data-testid="shared-messages" />,
}));

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-key",
  kind: "cloud",
};

const ORG_ID = "org-1";
const CREATOR_ID = "user-1";
const CONVERSATION_ID = "conv-automation-1";

function makeSharedConversation(): SharedConversationData {
  return {
    id: CONVERSATION_ID,
    created_by_user_id: CREATOR_ID,
    selected_repository: "OpenHands/agent-canvas",
    selected_branch: "main",
    git_provider: "github",
    title: "Automated code review",
    pr_number: [1688],
    llm_model: "openhands/claude-haiku-4-5-20251001",
    metrics: null,
    parent_conversation_id: null,
    sub_conversation_ids: [],
    created_at: "2026-09-24T12:00:00.000Z",
    updated_at: "2026-09-24T12:30:00.000Z",
  };
}

function renderViewer() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <MemoryRouter initialEntries={[`/shared/conversations/${CONVERSATION_ID}`]}>
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>
          <Routes>
            <Route
              path="/shared/conversations/:conversationId"
              element={<SharedConversation />}
            />
          </Routes>
        </ActiveBackendProvider>
      </QueryClientProvider>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([cloudBackend]);
  setActiveSelection({ backendId: cloudBackend.id, orgId: ORG_ID });
  vi.mocked(getCloudSharedConversation).mockReset();
  vi.mocked(searchCloudSharedEvents).mockReset();
  vi.mocked(getCloudOrganizationMember).mockReset();
  vi.mocked(getCloudSharedConversation).mockResolvedValue(
    makeSharedConversation(),
  );
  vi.mocked(searchCloudSharedEvents).mockResolvedValue({
    items: [],
    next_page_id: null,
  });
  vi.mocked(getCloudOrganizationMember).mockResolvedValue({
    org_id: ORG_ID,
    user_id: CREATOR_ID,
    email: "creator@example.com",
  });
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("shared conversation viewer — cloud backend", () => {
  it("loads the conversation through the cloud backend and shows the creator's email", async () => {
    // Arrange / Act
    renderViewer();

    // Assert
    expect(
      await screen.findByText(
        `${I18nKey.CONVERSATION$CREATED_BY}: creator@example.com`,
      ),
    ).toBeInTheDocument();
    expect(getCloudSharedConversation).toHaveBeenCalledWith(CONVERSATION_ID);
    expect(searchCloudSharedEvents).toHaveBeenCalledWith(
      expect.objectContaining({ conversationId: CONVERSATION_ID }),
    );
    expect(getCloudOrganizationMember).toHaveBeenCalledWith(
      ORG_ID,
      CREATOR_ID,
      expect.objectContaining({ id: cloudBackend.id }),
    );
  });

  it("falls back to the creator's user id when the member lookup fails", async () => {
    // Arrange — the creator belongs to an org the viewer is not browsing.
    vi.mocked(getCloudOrganizationMember).mockRejectedValue(
      new Error("Member not found"),
    );

    // Act
    renderViewer();

    // Assert
    expect(
      await screen.findByText(
        `${I18nKey.CONVERSATION$CREATED_BY}: ${CREATOR_ID}`,
      ),
    ).toBeInTheDocument();
  });
});
