import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createRoutesStub } from "react-router";
import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { I18nextProvider } from "react-i18next";
import i18n from "i18next";
import { NavigationProvider } from "#/context/navigation-context";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { describe, expect, it, vi } from "vitest";
import {
  setupConversationPanelTest,
  createMockConversation,
  cloudBackend,
  onCloseMock,
} from "./conversation-panel-test-utils";
import { ConversationPanel } from "#/components/features/conversation-panel/conversation-panel";
import { usePinnedConversationsStore } from "#/stores/pinned-conversations-store";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";

vi.mock("#/hooks/mutation/use-unified-stop-conversation", () => ({
  useUnifiedPauseConversation: () => ({ mutate: vi.fn() }),
}));
vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: vi.fn(),
  displayErrorToast: vi.fn(),
  TOAST_OPTIONS: {},
}));

const ORG_ID = "org-2";

const renderCloudPanel = () => {
  setRegisteredBackends([cloudBackend]);
  setActiveSelection({ backendId: cloudBackend.id, orgId: ORG_ID });

  const RouterStub = createRoutesStub([
    {
      Component: () => <ConversationPanel onClose={onCloseMock} />,
      path: "/",
    },
    { Component: () => null, path: "/conversations/:conversationId" },
  ]);
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <I18nextProvider i18n={i18n}>
        <ActiveBackendProvider>
          <NavigationProvider
            value={{
              currentPath: "/",
              conversationId: null,
              isNavigating: false,
              navigate: vi.fn(),
            }}
          >
            <RouterStub />
          </NavigationProvider>
        </ActiveBackendProvider>
      </I18nextProvider>
    </QueryClientProvider>,
  );
};

const unpinned = (id: string, title: string) =>
  createMockConversation({ id, title, pinned_at: null });

const pinned = (id: string, title: string, pinnedAt: string) =>
  createMockConversation({ id, title, pinned_at: pinnedAt });

const serveConversations = (
  listed: AppConversation[],
  pinnedOnServer: AppConversation[] = [],
) => {
  vi.spyOn(
    AgentServerConversationService,
    "searchConversations",
  ).mockResolvedValue({ items: listed, next_page_id: null });
  let serverPins = [...pinnedOnServer];
  vi.spyOn(
    AgentServerConversationService,
    "searchPinnedConversations",
  ).mockImplementation(async () => serverPins);
  return vi
    .spyOn(AgentServerConversationService, "updateConversationPinned")
    .mockImplementation(async (id, isPinned) => {
      const conversation = listed.find((c) => c.id === id);
      serverPins = isPinned
        ? [
            { ...conversation!, pinned_at: new Date().toISOString() },
            ...serverPins,
          ]
        : serverPins.filter((c) => c.id !== id);
      return createMockConversation();
    });
};

describe("ConversationPanel pins on a Cloud server that supports them", () => {
  setupConversationPanelTest();

  it("lists server pins newest first, including ones outside the loaded page", async () => {
    serveConversations(
      [unpinned("recent", "Recent")],
      [
        pinned("newer-pin", "Newer pin", "2026-10-02T00:00:00Z"),
        pinned("older-pin", "Older pin", "2026-10-01T00:00:00Z"),
      ],
    );

    renderCloudPanel();

    const section = await screen.findByTestId(
      "conversation-panel-pinned-section",
    );
    const titles = within(section)
      .getAllByTestId("conversation-card")
      .map((card) => card.textContent);
    expect(titles).toHaveLength(2);
    expect(titles[0]).toContain("Newer pin");
    expect(titles[1]).toContain("Older pin");
  });

  it("pins a conversation on the server and moves it to the pinned section", async () => {
    const user = userEvent.setup();
    const updatePinned = serveConversations([unpinned("a", "Alpha")]);

    renderCloudPanel();

    await user.hover(await screen.findByTestId("conversation-card"));
    await user.click(screen.getByTestId("conversation-pin-toggle-a"));

    expect(updatePinned).toHaveBeenCalledWith("a", true);
    const section = await screen.findByTestId(
      "conversation-panel-pinned-section",
    );
    expect(within(section).getByText("Alpha")).toBeInTheDocument();
    expect(usePinnedConversationsStore.getState().pinsByBackendId).toEqual({});
  });

  it("unpins a server-pinned conversation", async () => {
    const user = userEvent.setup();
    const updatePinned = serveConversations(
      [pinned("a", "Alpha", "2026-10-01T00:00:00Z")],
      [pinned("a", "Alpha", "2026-10-01T00:00:00Z")],
    );

    renderCloudPanel();

    const section = await screen.findByTestId(
      "conversation-panel-pinned-section",
    );
    await user.click(within(section).getByTestId("conversation-pin-toggle-a"));

    expect(updatePinned).toHaveBeenCalledWith("a", false);
    await waitFor(() =>
      expect(
        screen.queryByTestId("conversation-panel-pinned-section"),
      ).not.toBeInTheDocument(),
    );
  });

  it("copies pins saved in the browser to the server, oldest first, and clears them", async () => {
    const updatePinned = serveConversations([
      unpinned("new", "New"),
      unpinned("old", "Old"),
    ]);
    vi.spyOn(
      AgentServerConversationService,
      "batchGetAppConversations",
    ).mockResolvedValue([unpinned("new", "New"), null, unpinned("old", "Old")]);
    usePinnedConversationsStore.setState({
      pinsByBackendId: { [cloudBackend.id]: ["new", "deleted", "old"] },
    });

    renderCloudPanel();

    await waitFor(() =>
      expect(
        usePinnedConversationsStore.getState().pinsByBackendId[cloudBackend.id],
      ).toEqual([]),
    );
    expect(updatePinned.mock.calls).toEqual([
      ["old", true],
      ["new", true],
    ]);
  });
});

describe("ConversationPanel pins on a Cloud server without pin support", () => {
  setupConversationPanelTest();

  it("keeps pinning in the browser and never calls the pin endpoint", async () => {
    const user = userEvent.setup();
    const updatePinned = serveConversations([
      createMockConversation({ id: "a", title: "Alpha" }),
    ]);

    renderCloudPanel();

    await user.hover(await screen.findByTestId("conversation-card"));
    await user.click(screen.getByTestId("conversation-pin-toggle-a"));

    expect(
      usePinnedConversationsStore.getState().pinsByBackendId[cloudBackend.id],
    ).toEqual(["a"]);
    expect(updatePinned).not.toHaveBeenCalled();
    expect(
      await screen.findByTestId("conversation-panel-pinned-section"),
    ).toBeInTheDocument();
  });
});
