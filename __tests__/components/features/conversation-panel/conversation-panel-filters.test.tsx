import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import {
  setupConversationPanelTest,
  createMockConversation,
  mockConversations,
  openAdvancedOptions,
  renderConversationPanel,
} from "./conversation-panel-test-utils";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { ExecutionStatus } from "#/types/agent-server/core";
import { useConversationPanelPreferencesStore } from "#/stores/conversation-panel-preferences-store";
import { useArchivedConversationsStore } from "#/stores/archived-conversations-store";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

const mockStopConversationMutate = vi.fn();
vi.mock("#/hooks/mutation/use-unified-stop-conversation", () => ({
  useUnifiedPauseConversation: () => ({ mutate: mockStopConversationMutate }),
}));
vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: vi.fn(),
  displayErrorToast: vi.fn(),
  TOAST_OPTIONS: {},
}));

describe("ConversationPanel filters and pagination", () => {
  setupConversationPanelTest();

  it("scopes the list to the automation filter mode across hide and only", async () => {
    // Arrange: two manual conversations plus one automation run recognized
    // by its tags (local backend) and one by its trigger (cloud backend).
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [
        createMockConversation({ id: "1", title: "Manual 1" }),
        createMockConversation({ id: "2", title: "Manual 2" }),
        createMockConversation({
          id: "3",
          title: "Tagged Run",
          tags: { automationname: "Nightly Audit", automationtrigger: "cron" },
        }),
        createMockConversation({
          id: "4",
          title: "Cloud Run",
          trigger: "automation",
        }),
      ],
      next_page_id: null,
    });
    useConversationPanelPreferencesStore.setState({
      automationFilterMode: "hide-automations",
    });

    // Act + Assert: hide mode keeps only the manual conversations.
    renderConversationPanel();
    const cards = await screen.findAllByTestId("conversation-card");
    expect(cards).toHaveLength(2);
    expect(screen.queryByText("Tagged Run")).not.toBeInTheDocument();
    expect(screen.queryByText("Cloud Run")).not.toBeInTheDocument();

    // Act + Assert: only mode inverts the scope.
    act(() => {
      useConversationPanelPreferencesStore.setState({
        automationFilterMode: "only-automations",
      });
    });
    expect(await screen.findByText("Tagged Run")).toBeInTheDocument();
    expect(screen.getByText("Cloud Run")).toBeInTheDocument();
    expect(screen.queryByText("Manual 1")).not.toBeInTheDocument();
  });

  it("filters from the persistent filter bar and couples automation chips to the mode", async () => {
    // Arrange: one tagged manual conversation, one untagged, one automation
    // run (recognized by its automation tags).
    const user = userEvent.setup();
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [
        createMockConversation({
          id: "1",
          title: "Manual 1",
          tags: { project: "vault" },
        }),
        createMockConversation({ id: "2", title: "Manual 2" }),
        createMockConversation({
          id: "3",
          title: "Nightly Run",
          tags: { automationname: "Nightly Audit", automationtrigger: "cron" },
        }),
      ],
      next_page_id: null,
    });

    renderConversationPanel();

    // Selecting a facet in the layouts menu narrows the list.
    await user.click(screen.getByTestId("conversation-layouts-toggle"));
    await user.click(screen.getByTestId("tag-filters-section"));
    await user.click(screen.getByTestId("tag-facet-row-project=vault"));
    expect(await screen.findByText("Manual 1")).toBeInTheDocument();
    expect(screen.queryByText("Manual 2")).not.toBeInTheDocument();
    expect(
      useConversationPanelPreferencesStore.getState().selectedTagFacets,
    ).toEqual(["project=vault"]);

    // Deselecting the facet restores the full list.
    await user.click(screen.getByTestId("tag-facet-row-project=vault"));
    expect(await screen.findByText("Manual 2")).toBeInTheDocument();

    // The automation scope lives in the Advanced options modal.
    await user.click(screen.getByTestId("advanced-options-row"));
    await user.click(screen.getByTestId("automation-filter-only"));
    expect(
      useConversationPanelPreferencesStore.getState().automationFilterMode,
    ).toBe("only-automations");
    expect(await screen.findByText("Nightly Run")).toBeInTheDocument();
    expect(screen.queryByText("Manual 1")).not.toBeInTheDocument();
  });

  it("keeps an active tag filter visible outside the layouts menu", async () => {
    const user = userEvent.setup();
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [
        createMockConversation({
          id: "1",
          title: "Manual 1",
          tags: { project: "vault" },
        }),
        createMockConversation({ id: "2", title: "Manual 2" }),
      ],
      next_page_id: null,
    });

    renderConversationPanel();

    // Nothing to announce until something is actually filtering.
    expect(
      screen.queryByTestId("conversation-active-tag-filters"),
    ).not.toBeInTheDocument();

    await user.click(screen.getByTestId("conversation-layouts-toggle"));
    await user.click(screen.getByTestId("tag-filters-section"));
    await user.click(screen.getByTestId("tag-facet-row-project=vault"));
    await user.click(screen.getByTestId("conversation-layouts-toggle"));

    // With the menu closed, the strip is the only thing on screen that says
    // where Manual 2 went — the facet checkmark is two levels inside a menu.
    expect(
      await screen.findByTestId("active-tag-filter-project=vault"),
    ).toBeInTheDocument();
    expect(screen.queryByText("Manual 2")).not.toBeInTheDocument();

    // And the way back out is on the strip too, not buried with the facets.
    await user.click(screen.getByTestId("clear-tag-filters"));
    expect(await screen.findByText("Manual 2")).toBeInTheDocument();
    expect(
      screen.queryByTestId("conversation-active-tag-filters"),
    ).not.toBeInTheDocument();
  });

  it("keeps a persisted automation-name filter both reachable and visible", async () => {
    // `selectedAutomationNames` is persisted and narrows the list on its own,
    // so it needs a control that can see and undo it — otherwise a reload
    // hides conversations with nothing on screen to say why.
    const user = userEvent.setup();
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [
        createMockConversation({
          id: "1",
          title: "Nightly Run",
          tags: { automationid: "a-1", automationname: "Nightly Audit" },
        }),
        createMockConversation({
          id: "2",
          title: "Weekly Run",
          tags: { automationid: "a-2", automationname: "Weekly Sweep" },
        }),
      ],
      next_page_id: null,
    });

    renderConversationPanel();
    expect(await screen.findByText("Nightly Run")).toBeInTheDocument();

    // The name rows live in the advanced-options modal, under the scope.
    await user.click(screen.getByTestId("conversation-layouts-toggle"));
    await user.click(screen.getByTestId("advanced-options-row"));
    await user.click(screen.getByTestId("automation-filter-only"));
    await user.click(
      await screen.findByTestId("automation-name-row-Nightly Audit"),
    );
    await user.click(screen.getByTestId("advanced-options-close"));

    expect(await screen.findByText("Nightly Run")).toBeInTheDocument();
    expect(screen.queryByText("Weekly Run")).not.toBeInTheDocument();

    // With every menu closed, the strip is the only thing naming the
    // narrowing — and the way back out.
    const chip = await screen.findByTestId(
      "active-automation-filter-Nightly Audit",
    );
    expect(chip).toHaveTextContent("Nightly Audit");

    await user.click(chip);
    expect(await screen.findByText("Weekly Run")).toBeInTheDocument();
    expect(
      useConversationPanelPreferencesStore.getState().selectedAutomationNames,
    ).toEqual([]);
  });

  it("clears tag chips from the cards when the Tag chips preference is off", async () => {
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [
        createMockConversation({
          id: "1",
          title: "Tagged 1",
          tags: { project: "vault" },
        }),
      ],
      next_page_id: null,
    });
    useConversationPanelPreferencesStore.setState({ showTagsMetadata: true });

    renderConversationPanel();

    expect(
      await screen.findByTestId("conversation-card-tag-chip"),
    ).toBeInTheDocument();

    act(() => {
      useConversationPanelPreferencesStore.setState({
        showTagsMetadata: false,
      });
    });

    await waitFor(() => {
      expect(
        screen.queryByTestId("conversation-card-tag-chip"),
      ).not.toBeInTheDocument();
    });
  });

  it("keeps load more reachable with the filtered empty message when the automation filter hides every loaded conversation", async () => {
    // Arrange: page 1 holds only an automation run (hidden by the active
    // filter); a manual conversation sits on page 2.
    const user = userEvent.setup();
    vi.spyOn(AgentServerConversationService, "searchConversations")
      .mockResolvedValueOnce({
        items: [
          createMockConversation({
            id: "run",
            title: "Tagged Run",
            tags: { automationrunid: "run-1" },
          }),
        ],
        next_page_id: "page-2",
      })
      .mockResolvedValueOnce({
        items: [createMockConversation({ id: "manual", title: "Manual 1" })],
        next_page_id: null,
      });
    useConversationPanelPreferencesStore.setState({
      automationFilterMode: "hide-automations",
    });

    renderConversationPanel();

    // Assert: the filter-specific empty message shows and load more stays
    // reachable.
    await screen.findByText("CONVERSATION_PANEL$NO_AUTOMATION_MATCHES");
    const loadMore = await screen.findByTestId("load-more-conversations");

    // Act: fetching the next page surfaces the manual conversation.
    await user.click(loadMore);
    expect(await screen.findByText("Manual 1")).toBeInTheDocument();
    expect(
      screen.queryByText("CONVERSATION_PANEL$NO_AUTOMATION_MATCHES"),
    ).not.toBeInTheDocument();
  });

  it("orders the entire visible list by created_at when Created sort is selected, across the recent/older partition", async () => {
    // Arrange: three conversations whose `created_at` ordering diverges
    // from `updated_at` across the 1-hour partition cutoff. If the panel
    // honored only the within-bucket sort, "Old Touched" (created 10d ago
    // but touched 30m ago) would render in the recent bucket *above* the
    // "Mid Stale" / "Newest Stale" entries that live in the older bucket
    // but were created more recently. Created-sort must order the whole
    // visible list by `created_at`, not just within each partition.
    const now = Date.now();
    const isoMinutesAgo = (m: number) =>
      new Date(now - m * 60 * 1000).toISOString();
    const isoDaysAgo = (d: number) =>
      new Date(now - d * 24 * 60 * 60 * 1000).toISOString();

    useConversationPanelPreferencesStore.setState({
      conversationSort: "updated",
      organizeMode: "chronological",
      showOlderConversations: true,
    });

    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [
        createMockConversation({
          id: "old-touched",
          title: "Old Touched",
          created_at: isoDaysAgo(10),
          updated_at: isoMinutesAgo(30),
        }),
        createMockConversation({
          id: "newest-stale",
          title: "Newest Stale",
          created_at: isoDaysAgo(1),
          updated_at: isoDaysAgo(1),
        }),
        createMockConversation({
          id: "mid-stale",
          title: "Mid Stale",
          created_at: isoDaysAgo(3),
          updated_at: isoDaysAgo(3),
        }),
      ],
      next_page_id: null,
    });

    const user = userEvent.setup();
    renderConversationPanel();
    await screen.findByText("Old Touched");

    // Act: open the filter menu and switch sort to Created.
    await openAdvancedOptions(user);
    await user.click(
      screen.getByRole("menuitemradio", {
        name: /CONVERSATION_PANEL\$SORT_CREATED/,
      }),
    );

    // Assert: rendered cards are in strict created_at desc order across
    // the full visible list, regardless of which partition they came from.
    const cards = await screen.findAllByTestId("conversation-card");
    expect(cards.map((card) => card.textContent ?? "")).toEqual([
      expect.stringContaining("Newest Stale"),
      expect.stringContaining("Mid Stale"),
      expect.stringContaining("Old Touched"),
    ]);
  });

  describe("older conversations cutoff", () => {
    const recentIso = () => new Date().toISOString();
    const olderIso = () =>
      new Date(Date.now() - 8 * 24 * 60 * 60 * 1000).toISOString();

    it("shows conversations older than 1 week and includes a summary line", async () => {
      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [
          createMockConversation({
            id: "recent",
            title: "Recent",
            updated_at: recentIso(),
          }),
          createMockConversation({
            id: "old1",
            title: "Old 1",
            updated_at: olderIso(),
          }),
          createMockConversation({
            id: "old2",
            title: "Old 2",
            updated_at: olderIso(),
          }),
        ],
        next_page_id: null,
      });

      renderConversationPanel();

      const cards = await screen.findAllByTestId("conversation-card");
      expect(cards).toHaveLength(3);
      expect(screen.getByText("Recent")).toBeInTheDocument();
      expect(screen.getByText("Old 1")).toBeInTheDocument();
      expect(screen.getByText("Old 2")).toBeInTheDocument();

      const summary = screen.getByTestId("older-conversations-summary");
      expect(summary).toHaveTextContent("SIDEBAR$CONVERSATIONS");
      expect(
        within(summary).getByTestId("conversation-layouts-toggle"),
      ).toBeInTheDocument();
    });

    it("always renders the conversations header with the filter control", async () => {
      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [
          createMockConversation({
            id: "recent1",
            title: "Recent 1",
            updated_at: recentIso(),
          }),
          createMockConversation({
            id: "recent2",
            title: "Recent 2",
            updated_at: recentIso(),
          }),
        ],
        next_page_id: null,
      });

      renderConversationPanel();

      await screen.findAllByTestId("conversation-card");
      const summary = screen.getByTestId("older-conversations-summary");
      expect(summary).toBeInTheDocument();
      expect(
        within(summary).getByTestId("conversation-layouts-toggle"),
      ).toBeInTheDocument();
    });

    it("shows icons on hide and delete-all filter menu actions", async () => {
      const user = userEvent.setup();
      renderConversationPanel();

      // Delete all lives in the layouts menu itself.
      await user.click(screen.getByTestId("conversation-layouts-toggle"));
      const deleteAllRow = screen.getByTestId("delete-all-conversations");
      expect(deleteAllRow.querySelector("svg")).toBeInTheDocument();
      expect(deleteAllRow).toHaveClass("text-danger");
      expect(deleteAllRow).not.toHaveClass("text-[var(--oh-foreground)]");

      // The older-conversations toggle lives in the Advanced options modal.
      await user.click(screen.getByTestId("advanced-options-row"));
      const hideRow = await screen.findByTestId("toggle-older-conversations");
      expect(hideRow.querySelector("svg")).toBeInTheDocument();
      expect(hideRow).toHaveClass("group");
    });

    it("toggles older conversations visibility via the filter dropdown", async () => {
      const user = userEvent.setup();
      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [
          createMockConversation({
            id: "recent",
            title: "Recent",
            updated_at: recentIso(),
          }),
          createMockConversation({
            id: "old1",
            title: "Old 1",
            updated_at: olderIso(),
          }),
        ],
        next_page_id: null,
      });

      renderConversationPanel();

      let cards = await screen.findAllByTestId("conversation-card");
      expect(cards).toHaveLength(2);

      await openAdvancedOptions(user);
      let toggle = await screen.findByTestId("toggle-older-conversations");
      expect(toggle).toHaveAttribute("aria-checked", "false");
      await user.click(toggle);

      cards = await screen.findAllByTestId("conversation-card");
      expect(cards).toHaveLength(1);

      toggle = await screen.findByTestId("toggle-older-conversations");

      expect(toggle).toHaveAttribute("aria-checked", "true");
      await user.click(toggle);
      cards = await screen.findAllByTestId("conversation-card");
      expect(cards).toHaveLength(2);
    });

    it("keeps repo/branch metadata hidden by default and toggles it from the filter dropdown", async () => {
      const user = userEvent.setup();
      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [
          createMockConversation({
            id: "recent",
            title: "Recent",
            updated_at: recentIso(),
          }),
          createMockConversation({
            id: "old-with-repo",
            title: "Old With Repo",
            updated_at: olderIso(),
            selected_repository: "openhands/agent-canvas",
            selected_branch: "main",
            git_provider: "github",
          }),
        ],
        next_page_id: null,
      });

      renderConversationPanel();
      await screen.findByText("Old With Repo");

      expect(
        screen.queryByTestId("conversation-card-selected-repository"),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByTestId("conversation-card-selected-branch"),
      ).not.toBeInTheDocument();

      await openAdvancedOptions(user);
      await user.click(screen.getByTestId("toggle-repo-branch-metadata"));

      expect(
        await screen.findByTestId("conversation-card-selected-repository"),
      ).toHaveTextContent("openhands/agent-canvas");
      expect(
        await screen.findByTestId("conversation-card-selected-branch"),
      ).toHaveTextContent("main");
    });

    it("delete-all is enabled when no conversations are older than the cutoff and deletes every loaded conversation", async () => {
      const user = userEvent.setup();
      const deleteSpy = vi
        .spyOn(AgentServerConversationService, "deleteConversation")
        .mockResolvedValue();

      // Fixture: only recent conversations (none older than 1h). Before the
      // fix the "Delete all" button was disabled in this state.
      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [
          createMockConversation({
            id: "recent-1",
            title: "Recent 1",
            updated_at: recentIso(),
          }),
          createMockConversation({
            id: "recent-2",
            title: "Recent 2",
            updated_at: recentIso(),
          }),
        ],
        next_page_id: null,
      });

      renderConversationPanel();
      await screen.findAllByTestId("conversation-card");

      await user.click(screen.getByTestId("conversation-layouts-toggle"));
      const deleteAllButton = await screen.findByTestId(
        "delete-all-conversations",
      );
      expect(deleteAllButton).toBeEnabled();

      await user.click(deleteAllButton);
      await user.click(await screen.findByRole("button", { name: /confirm/i }));

      await waitFor(() => {
        expect(deleteSpy).toHaveBeenCalledTimes(2);
      });
      expect(deleteSpy).toHaveBeenCalledWith("recent-1");
      expect(deleteSpy).toHaveBeenCalledWith("recent-2");
    });

    it("delete-all still deletes hidden archived conversations", async () => {
      // "Show archived" only controls rendering. Delete all must keep using
      // the full loaded collection so archived server-side conversations are
      // not silently left behind (or the action disabled when every loaded
      // row is archived).
      const user = userEvent.setup();
      const deleteSpy = vi
        .spyOn(AgentServerConversationService, "deleteConversation")
        .mockResolvedValue();

      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [
          createMockConversation({
            id: "visible-1",
            title: "Visible 1",
            updated_at: recentIso(),
          }),
          createMockConversation({
            id: "archived-1",
            title: "Archived 1",
            updated_at: recentIso(),
          }),
          createMockConversation({
            id: "archived-2",
            title: "Archived 2",
            updated_at: recentIso(),
          }),
        ],
        next_page_id: null,
      });

      useArchivedConversationsStore.setState({
        archivesByBackendId: {
          "default-local": ["archived-1", "archived-2"],
        },
      });
      useConversationPanelPreferencesStore.setState({
        showArchivedConversations: false,
      });

      renderConversationPanel();
      const cards = await screen.findAllByTestId("conversation-card");
      expect(cards).toHaveLength(1);
      expect(screen.getByText("Visible 1")).toBeInTheDocument();
      expect(screen.queryByText("Archived 1")).not.toBeInTheDocument();

      await user.click(screen.getByTestId("conversation-layouts-toggle"));
      const deleteAllButton = await screen.findByTestId(
        "delete-all-conversations",
      );
      expect(deleteAllButton).toBeEnabled();

      await user.click(deleteAllButton);
      expect(
        await screen.findByText(/CONVERSATION\$CONFIRM_DELETE_ALL_DESC/),
      ).toBeInTheDocument();
      await user.click(await screen.findByRole("button", { name: /confirm/i }));

      await waitFor(() => {
        expect(deleteSpy).toHaveBeenCalledTimes(3);
      });
      expect(deleteSpy).toHaveBeenCalledWith("visible-1");
      expect(deleteSpy).toHaveBeenCalledWith("archived-1");
      expect(deleteSpy).toHaveBeenCalledWith("archived-2");
    });

    it("navigates away after the active conversation is deleted successfully even when another deletion fails", async () => {
      const user = userEvent.setup();
      const navigate = vi.fn();
      const deleteSpy = vi
        .spyOn(AgentServerConversationService, "deleteConversation")
        .mockImplementation(async (conversationId: string) => {
          if (conversationId === "old2") {
            throw new Error("delete failed");
          }
        });

      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [
          createMockConversation({
            id: "recent",
            title: "Recent",
            updated_at: recentIso(),
          }),
          createMockConversation({
            id: "old1",
            title: "Old 1",
            updated_at: olderIso(),
          }),
          createMockConversation({
            id: "old2",
            title: "Old 2",
            updated_at: olderIso(),
          }),
        ],
        next_page_id: null,
      });

      // Active conversation is "old1" — it is among the conversations that
      // get deleted successfully, so we should navigate away.
      renderConversationPanel({
        navigation: { conversationId: "old1", navigate },
      });
      await screen.findAllByTestId("conversation-card");

      await user.click(screen.getByTestId("conversation-layouts-toggle"));
      await user.click(screen.getByTestId("delete-all-conversations"));
      await user.click(await screen.findByRole("button", { name: /confirm/i }));

      await waitFor(() => {
        expect(deleteSpy).toHaveBeenCalledTimes(3);
      });
      expect(displayErrorToast).toHaveBeenCalledWith(
        "1 conversation could not be deleted.",
      );
      expect(navigate).toHaveBeenCalledWith("/conversations");
    });

    it("does not navigate away when the active conversation fails to delete", async () => {
      const user = userEvent.setup();
      const navigate = vi.fn();
      const deleteSpy = vi
        .spyOn(AgentServerConversationService, "deleteConversation")
        .mockImplementation(async (conversationId: string) => {
          if (conversationId === "old1") {
            throw new Error("delete failed");
          }
        });

      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [
          createMockConversation({
            id: "recent",
            title: "Recent",
            updated_at: recentIso(),
          }),
          createMockConversation({
            id: "old1",
            title: "Old 1",
            updated_at: olderIso(),
          }),
          createMockConversation({
            id: "old2",
            title: "Old 2",
            updated_at: olderIso(),
          }),
        ],
        next_page_id: null,
      });

      // Active conversation is "old1" — its deletion fails, so we must
      // not navigate away from it.
      renderConversationPanel({
        navigation: { conversationId: "old1", navigate },
      });
      await screen.findAllByTestId("conversation-card");

      await user.click(screen.getByTestId("conversation-layouts-toggle"));
      await user.click(screen.getByTestId("delete-all-conversations"));
      await user.click(await screen.findByRole("button", { name: /confirm/i }));

      await waitFor(() => {
        expect(deleteSpy).toHaveBeenCalledTimes(3);
      });
      expect(displayErrorToast).toHaveBeenCalledWith(
        "1 conversation could not be deleted.",
      );
      expect(navigate).not.toHaveBeenCalled();
    });
  });

  describe("load-more link", () => {
    const recentIso = () => new Date().toISOString();
    const olderIso = () =>
      new Date(Date.now() - 8 * 24 * 60 * 60 * 1000).toISOString();

    it("shows a load-more link when there is a next page and no older conversations are hidden", async () => {
      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [
          createMockConversation({
            id: "recent",
            title: "Recent",
            updated_at: recentIso(),
          }),
        ],
        next_page_id: "page-2",
      });

      renderConversationPanel();

      await screen.findAllByTestId("conversation-card");
      const loadMore = await screen.findByTestId("load-more-conversations");
      expect(loadMore).toHaveTextContent("CONVERSATION$LOAD_MORE");
    });

    it("hides the load-more link after older conversations are hidden from the filter dropdown", async () => {
      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [
          createMockConversation({
            id: "recent",
            title: "Recent",
            updated_at: recentIso(),
          }),
          createMockConversation({
            id: "old1",
            title: "Old 1",
            updated_at: olderIso(),
          }),
        ],
        next_page_id: "page-2",
      });

      renderConversationPanel();

      await screen.findAllByTestId("conversation-card");
      // Older conversations are visible by default, so load-more is visible.
      expect(screen.getByTestId("load-more-conversations")).toBeInTheDocument();

      // Hide older conversations via the filter dropdown.
      const user = userEvent.setup();
      await openAdvancedOptions(user);
      await user.click(screen.getByTestId("toggle-older-conversations"));

      // Older conversations are hidden → no load-more.
      expect(
        screen.queryByTestId("load-more-conversations"),
      ).not.toBeInTheDocument();

      // After showing older conversations again, the link reappears.
      await openAdvancedOptions(user);
      await user.click(screen.getByTestId("toggle-older-conversations"));
      expect(
        await screen.findByTestId("load-more-conversations"),
      ).toBeInTheDocument();
    });

    it("fetches the next page when the load-more link is clicked", async () => {
      const user = userEvent.setup();
      const searchSpy = vi
        .spyOn(AgentServerConversationService, "searchConversations")
        .mockResolvedValueOnce({
          items: [
            createMockConversation({
              id: "recent",
              title: "Recent",
              updated_at: recentIso(),
            }),
          ],
          next_page_id: "page-2",
        })
        .mockResolvedValueOnce({
          items: [
            createMockConversation({
              id: "page2-1",
              title: "Page 2 Conversation",
              updated_at: recentIso(),
            }),
          ],
          next_page_id: null,
        });

      renderConversationPanel();

      const loadMore = await screen.findByTestId("load-more-conversations");
      await user.click(loadMore);

      await waitFor(() => {
        expect(searchSpy).toHaveBeenCalledTimes(2);
      });

      // After the second page resolves, the link disappears (no more pages).
      await waitFor(() => {
        expect(
          screen.queryByTestId("load-more-conversations"),
        ).not.toBeInTheDocument();
      });
    });

    it("keeps collapsed folder previews stable while still exposing later-page conversations on expand", async () => {
      useConversationPanelPreferencesStore.setState({
        organizeMode: "grouped",
      });
      const noWorkspaceConversations = Array.from({ length: 6 }, (_, index) =>
        createMockConversation({
          id: `no-workspace-${index + 1}`,
          title: `No workspace ${index + 1}`,
        }),
      );
      const searchSpy = vi
        .spyOn(AgentServerConversationService, "searchConversations")
        .mockResolvedValueOnce({
          items: noWorkspaceConversations,
          next_page_id: "page-2",
        })
        .mockResolvedValueOnce({
          items: [
            createMockConversation({
              id: "no-workspace-7",
              title: "No workspace 7",
            }),
            createMockConversation({
              id: "no-workspace-8",
              title: "No workspace 8",
            }),
          ],
          next_page_id: "page-3",
        })
        .mockResolvedValueOnce({
          items: [
            createMockConversation({
              id: "alpha",
              title: "Alpha conversation",
              selected_workspace: "/workspace/alpha",
            }),
          ],
          next_page_id: null,
        });

      const user = userEvent.setup();
      renderConversationPanel();

      const noWorkspaceFolder = await screen.findByTestId(
        "thread-folder-__none_workspace",
      );
      expect(
        within(noWorkspaceFolder).getAllByTestId("conversation-card"),
      ).toHaveLength(5);
      expect(
        within(noWorkspaceFolder).queryByText("No workspace 7"),
      ).not.toBeInTheDocument();

      // A single click walks past the deepen-only page 2 (its rows are hidden
      // from the collapsed preview, so they are not success) and keeps paging
      // until the brand-new folder on page 3 is discovered.
      await user.click(screen.getByTestId("load-more-conversations"));
      await screen.findByTestId("thread-folder-ws--workspace-alpha");
      expect(searchSpy).toHaveBeenCalledTimes(3);
      expect(
        within(noWorkspaceFolder).getAllByTestId("conversation-card"),
      ).toHaveLength(5);
      expect(
        within(noWorkspaceFolder).queryByText("No workspace 7"),
      ).not.toBeInTheDocument();

      // Collapsed preview stays frozen; expanding reveals every loaded row.
      expect(
        within(noWorkspaceFolder).getAllByTestId("conversation-card"),
      ).toHaveLength(5);
      await user.click(
        within(noWorkspaceFolder).getByTestId(
          "thread-folder-view-more-__none_workspace",
        ),
      );
      expect(
        within(noWorkspaceFolder).getAllByTestId("conversation-card"),
      ).toHaveLength(8);
      expect(
        within(noWorkspaceFolder).getByText("No workspace 7"),
      ).toBeInTheDocument();
      expect(
        within(noWorkspaceFolder).getByText("No workspace 8"),
      ).toBeInTheDocument();
    });

    it("caps a grouped load-more click at three pages when no new folder appears", async () => {
      useConversationPanelPreferencesStore.setState({
        organizeMode: "grouped",
      });
      const deepenOnlyPage = (page: number, nextPageId: string) => ({
        items: Array.from({ length: 2 }, (_, index) =>
          createMockConversation({
            id: `no-workspace-p${page}-${index + 1}`,
            title: `No workspace p${page}-${index + 1}`,
          }),
        ),
        next_page_id: nextPageId,
      });
      // Exactly four pages are queued: the drive must consume the initial page
      // plus MAX_PAGES_PER_LOAD_MORE_CLICK more and stop — a fetch beyond the
      // cap would find no queued response and fail the test loudly.
      const searchSpy = vi
        .spyOn(AgentServerConversationService, "searchConversations")
        .mockResolvedValueOnce(deepenOnlyPage(1, "page-2"))
        .mockResolvedValueOnce(deepenOnlyPage(2, "page-3"))
        .mockResolvedValueOnce(deepenOnlyPage(3, "page-4"))
        .mockResolvedValueOnce(deepenOnlyPage(4, "page-5"));

      const user = userEvent.setup();
      renderConversationPanel();

      await screen.findByTestId("thread-folder-__none_workspace");
      expect(searchSpy).toHaveBeenCalledTimes(1);

      // Every remaining page only deepens the existing folder, so the driver
      // must stop at the per-click page cap instead of draining the cursor.
      await user.click(screen.getByTestId("load-more-conversations"));
      await waitFor(() => {
        expect(searchSpy).toHaveBeenCalledTimes(4);
      });
      // The drive has ended: the control is idle again and no further page
      // was requested beyond the cap.
      await waitFor(() => {
        expect(
          screen.getByTestId("load-more-conversations"),
        ).toBeInTheDocument();
      });
      expect(searchSpy).toHaveBeenCalledTimes(4);
    });

    it("force-includes the active conversation in a folder preview even when it arrived on a later page", async () => {
      useConversationPanelPreferencesStore.setState({
        organizeMode: "grouped",
      });
      vi.spyOn(AgentServerConversationService, "searchConversations")
        .mockResolvedValueOnce({
          items: Array.from({ length: 5 }, (_, index) =>
            createMockConversation({
              id: `alpha-${index + 1}`,
              title: `Alpha ${index + 1}`,
              selected_workspace: "/workspace/alpha",
            }),
          ),
          next_page_id: "page-2",
        })
        .mockResolvedValueOnce({
          items: [
            createMockConversation({
              id: "alpha-active",
              title: "Alpha Active",
              selected_workspace: "/workspace/alpha",
            }),
          ],
          next_page_id: null,
        });

      const user = userEvent.setup();
      renderConversationPanel({
        navigation: {
          conversationId: "alpha-active",
          currentPath: "/conversations/alpha-active",
        },
      });

      const alphaFolder = await screen.findByTestId(
        "thread-folder-ws--workspace-alpha",
      );
      expect(
        within(alphaFolder).queryByText("Alpha Active"),
      ).not.toBeInTheDocument();

      await user.click(screen.getByTestId("load-more-conversations"));

      await waitFor(() => {
        expect(
          within(alphaFolder).getByText("Alpha Active"),
        ).toBeInTheDocument();
      });
      // Still a collapsed preview (limit 5), not the full expanded list.
      expect(
        within(alphaFolder).getAllByTestId("conversation-card"),
      ).toHaveLength(5);
    });
  });
});
