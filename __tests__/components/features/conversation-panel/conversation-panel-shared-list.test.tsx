import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import {
  setupConversationPanelTest,
  cloudBackend,
  renderConversationPanel,
} from "./conversation-panel-test-utils";
import { ConversationPanel } from "#/components/features/conversation-panel/conversation-panel";
import { NavigationProvider } from "#/context/navigation-context";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { getCloudOrganizationMe } from "#/api/cloud/organization-service.api";
import { searchCloudSharedConversations } from "#/api/cloud/shared-conversation-service.api";

vi.mock("#/api/cloud/organization-service.api", async (importOriginal) => ({
  ...(await importOriginal<object>()),
  getCloudOrganizationMe: vi.fn(),
}));
vi.mock(
  "#/api/cloud/shared-conversation-service.api",
  async (importOriginal) => ({
    ...(await importOriginal<object>()),
    searchCloudSharedConversations: vi.fn(),
  }),
);

const TEAM_ORG_ID = "org-team";

const sharedConversation = (id: string, title: string) => ({
  id,
  title,
  created_by_user_id: "teammate",
  selected_repository: null,
  selected_branch: null,
  git_provider: null,
  pr_number: [],
  llm_model: null,
  metrics: null,
  parent_conversation_id: null,
  sub_conversation_ids: [],
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
});

// The caller is a member of the team org: their user id is not the org id.
const signInAsTeamOrgMember = () =>
  vi.mocked(getCloudOrganizationMe).mockResolvedValue({
    orgId: TEAM_ORG_ID,
    userId: "user-1",
    role: "member",
    permissions: null,
  });

const renderCloudPanel = (orgId: string) => {
  setRegisteredBackends([cloudBackend]);
  setActiveSelection({ backendId: cloudBackend.id, orgId });
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <ActiveBackendProvider>
        <NavigationProvider
          value={{
            currentPath: "/conversations",
            conversationId: null,
            navigate: vi.fn(),
            isNavigating: false,
          }}
        >
          <ConversationPanel />
        </NavigationProvider>
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
};

const selectSharedList = async (user: ReturnType<typeof userEvent.setup>) =>
  user.click(
    await screen.findByTestId("conversation-list-scope-option-shared"),
  );

describe("ConversationPanel shared list", () => {
  setupConversationPanelTest();

  it("lists the conversations shared with a team org when Shared is selected", async () => {
    // Arrange
    const user = userEvent.setup();
    signInAsTeamOrgMember();
    vi.mocked(searchCloudSharedConversations).mockResolvedValue({
      items: [sharedConversation("shared-1", "Teammate handoff")],
      next_page_id: null,
    });
    renderCloudPanel(TEAM_ORG_ID);

    // Act
    await selectSharedList(user);

    // Assert
    const row = await screen.findByRole("link", { name: /Teammate handoff/ });
    expect(row).toHaveAttribute(
      "href",
      expect.stringContaining("/conversations/shared-1"),
    );
  });

  it("loads the next page of shared conversations on request", async () => {
    // Arrange
    const user = userEvent.setup();
    signInAsTeamOrgMember();
    vi.mocked(searchCloudSharedConversations).mockImplementation(
      async ({ pageId }) =>
        pageId === "20"
          ? {
              items: [sharedConversation("shared-2", "Older handoff")],
              next_page_id: null,
            }
          : {
              items: [sharedConversation("shared-1", "Teammate handoff")],
              next_page_id: "20",
            },
    );
    renderCloudPanel(TEAM_ORG_ID);
    await selectSharedList(user);

    // Act
    await user.click(
      await screen.findByTestId("load-more-shared-conversations"),
    );

    // Assert
    expect(await screen.findByText("Older handoff")).toBeInTheDocument();
    expect(screen.getByText("Teammate handoff")).toBeInTheDocument();
  });

  it("says no conversations were found when nothing is shared with the org", async () => {
    // Arrange
    const user = userEvent.setup();
    signInAsTeamOrgMember();
    vi.mocked(searchCloudSharedConversations).mockResolvedValue({
      items: [],
      next_page_id: null,
    });
    renderCloudPanel(TEAM_ORG_ID);

    // Act
    await selectSharedList(user);

    // Assert
    expect(
      await screen.findByTestId("shared-conversation-list-empty-state"),
    ).toBeInTheDocument();
  });

  it("does not offer the private list's layouts menu while Shared is selected", async () => {
    // Arrange
    const user = userEvent.setup();
    signInAsTeamOrgMember();
    vi.mocked(searchCloudSharedConversations).mockResolvedValue({
      items: [],
      next_page_id: null,
    });
    renderCloudPanel(TEAM_ORG_ID);

    // Act
    await selectSharedList(user);

    // Assert
    expect(
      screen.queryByTestId("conversation-layouts-toggle"),
    ).not.toBeInTheDocument();
  });

  it("goes back to the user's own conversations when Private is selected again", async () => {
    // Arrange
    const user = userEvent.setup();
    signInAsTeamOrgMember();
    vi.mocked(searchCloudSharedConversations).mockResolvedValue({
      items: [sharedConversation("shared-1", "Teammate handoff")],
      next_page_id: null,
    });
    renderCloudPanel(TEAM_ORG_ID);
    await selectSharedList(user);
    await screen.findByText("Teammate handoff");

    // Act
    await user.click(
      screen.getByTestId("conversation-list-scope-option-private"),
    );

    // Assert
    expect(screen.queryByText("Teammate handoff")).not.toBeInTheDocument();
    expect(screen.getByText("Conversation 1")).toBeInTheDocument();
    expect(
      screen.getByTestId("conversation-layouts-toggle"),
    ).toBeInTheDocument();
  });

  it("does not offer the Shared list in a personal workspace", async () => {
    // Arrange — a personal workspace is the org whose id is the user's id.
    vi.mocked(getCloudOrganizationMe).mockResolvedValue({
      orgId: "org-personal",
      userId: "org-personal",
      role: "owner",
      permissions: null,
    });

    // Act
    renderCloudPanel("org-personal");
    await screen.findByText("Conversation 1");
    await waitFor(() => expect(getCloudOrganizationMe).toHaveBeenCalled());

    // Assert
    expect(
      screen.queryByTestId("conversation-list-scope"),
    ).not.toBeInTheDocument();
  });

  it("does not offer the Shared list on a local backend", async () => {
    // Act
    renderConversationPanel();
    await screen.findByText("Conversation 1");

    // Assert
    expect(
      screen.queryByTestId("conversation-list-scope"),
    ).not.toBeInTheDocument();
  });
});
