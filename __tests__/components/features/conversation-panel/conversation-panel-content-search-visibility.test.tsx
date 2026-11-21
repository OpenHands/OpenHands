import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { renderWithProviders } from "test-utils";
import {
  RouterStub,
  cloudBackend,
  setupConversationPanelTest,
} from "./conversation-panel-test-utils";
import {
  isContentSearchEnabled,
  searchConversationContent,
} from "#/api/conversation-service/content-search.api";
import type { Backend } from "#/api/backend-registry/types";

vi.mock("#/api/conversation-service/content-search.api", () => ({
  searchConversationContent: vi.fn(),
  isContentSearchEnabled: vi.fn(),
}));

const localBackend: Backend = {
  id: "local-dev",
  name: "Local",
  host: "http://localhost:18000",
  apiKey: "local-key",
  kind: "local",
} as const;

beforeEach(() => {
  // The box only renders when the availability probe resolves true.
  vi.mocked(isContentSearchEnabled).mockResolvedValue(true);
});

describe("ConversationPanel content search visibility", () => {
  setupConversationPanelTest();

  it("does not render the search box on a cloud backend", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-2" });
    renderWithProviders(
      <ActiveBackendProvider>
        <RouterStub />
      </ActiveBackendProvider>,
    );

    // The panel settles (cards render) while the search box stays absent.
    await screen.findAllByTestId("conversation-card");
    expect(
      screen.queryByTestId("conversation-content-search"),
    ).not.toBeInTheDocument();
  });

  it("does not fire a content-search query on a cloud backend", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-2" });
    renderWithProviders(
      <ActiveBackendProvider>
        <RouterStub />
      </ActiveBackendProvider>,
    );

    await screen.findAllByTestId("conversation-card");
    expect(searchConversationContent).not.toHaveBeenCalled();
  });

  it("renders the search box on a local backend", async () => {
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    renderWithProviders(
      <ActiveBackendProvider>
        <RouterStub />
      </ActiveBackendProvider>,
    );

    expect(
      await screen.findByTestId("conversation-content-search"),
    ).toBeInTheDocument();
  });
});
