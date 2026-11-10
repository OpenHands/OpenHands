import { http, HttpResponse } from "msw";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import { renderWithProviders } from "test-utils";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { ChatAddFileButton } from "#/components/features/chat/chat-add-file-button";
import { useConversationConfigurationCapability } from "#/hooks/query/use-conversation-configuration-capability";
import { server } from "#/mocks/node";

function renderToolbar(capabilities: string[], onProbe?: () => void) {
  server.use(
    http.get("*/server_info", () => {
      onProbe?.();
      return HttpResponse.json({
        version: "1.61.0",
        uptime: 1,
        idle_time: 0,
        capabilities,
      });
    }),
  );
  return renderWithProviders(
    <ActiveBackendProvider>
      <ChatAddFileButton handleFileIconClick={() => {}} />
    </ActiveBackendProvider>,
    { navigation: { conversationId: "2" } },
  );
}

function CapabilityProbe() {
  const { data } = useConversationConfigurationCapability(true);
  return (
    <span
      data-testid="configuration-capability"
      data-supported={data === true}
    />
  );
}

afterEach(() => __resetActiveStoreForTests());

describe("conversation configuration toolbar", () => {
  it("opens the scoped modal only when the local server advertises the capability", async () => {
    const user = userEvent.setup();
    renderToolbar(["conversation_configuration_additions_v1"]);
    await screen.findByTestId("chat-plus-button");
    await user.click(screen.getByTestId("chat-plus-button"));
    const action = await screen.findByTestId(
      "conversation-configuration-button",
    );
    await user.click(action);
    expect(
      await screen.findByTestId("conversation-configuration-modal"),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("tools-context-menu")).not.toBeInTheDocument();
  });

  it("offers an inline marketplace only when the local server advertises scoped MCP installs", async () => {
    const user = userEvent.setup();
    let probes = 0;
    renderToolbar(
      [
        "conversation_configuration_additions_v1",
        "conversation_mcp_install_v1",
      ],
      () => {
        probes += 1;
      },
    );
    await user.click(await screen.findByTestId("chat-plus-button"));
    await user.click(
      await screen.findByTestId("conversation-configuration-button"),
    );
    expect(
      await screen.findByTestId("mcp-marketplace-section"),
    ).toBeInTheDocument();
    expect(screen.queryByText("MCP$VIEW_MARKETPLACE")).not.toBeInTheDocument();
    expect(probes).toBe(1);
  });

  it("keeps unsupported local servers free of the new menu action", async () => {
    const user = userEvent.setup();
    let probes = 0;
    renderToolbar([], () => {
      probes += 1;
    });
    await waitFor(() => expect(probes).toBe(1));
    await user.click(screen.getByTestId("chat-plus-button"));
    expect(
      screen.getByTestId("add-files-and-images-button"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("conversation-configuration-button"),
    ).not.toBeInTheDocument();
  });

  it("does not even probe the local agent-server capability for a cloud backend", async () => {
    let probes = 0;
    server.use(
      http.get("*/server_info", () => {
        probes += 1;
        return HttpResponse.json({
          capabilities: ["conversation_configuration_additions_v1"],
        });
      }),
    );
    setRegisteredBackends([
      {
        id: "cloud-test",
        name: "Cloud",
        kind: "cloud",
        host: "https://example.com",
        apiKey: "",
      },
    ]);
    setActiveSelection({ backendId: "cloud-test", orgId: "org-test" });
    renderWithProviders(
      <ActiveBackendProvider>
        <CapabilityProbe />
      </ActiveBackendProvider>,
    );
    expect(screen.getByTestId("configuration-capability")).toHaveAttribute(
      "data-supported",
      "false",
    );
    expect(probes).toBe(0);
  });
});
