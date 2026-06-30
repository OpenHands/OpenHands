import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router";
import { I18nextProvider, initReactI18next } from "react-i18next";
import i18n from "i18next";
import { McpUnavailableBanner } from "#/components/features/chat/mcp-unavailable-banner";
import { useMcpConversationHealth } from "#/hooks/query/use-mcp-conversation-health";
import { useMcpWarningDismissStore } from "#/stores/mcp-warning-dismiss-store";
import { I18nKey } from "#/i18n/declaration";

vi.mock("#/hooks/query/use-mcp-conversation-health");

i18n.use(initReactI18next).init({
  lng: "en",
  fallbackLng: "en",
  resources: {
    en: {
      translation: {
        [I18nKey.CONVERSATION$MCP_UNAVAILABLE_SINGLE]:
          'MCP server "{{serverName}}" is unavailable: {{detail}}. <settingsLink>Check MCP settings</settingsLink>.',
        [I18nKey.CONVERSATION$MCP_UNAVAILABLE_MULTIPLE]:
          "{{count}} MCP servers are unavailable.",
        [I18nKey.CONVERSATION$MCP_UNAVAILABLE_SERVER_LINE]:
          "{{serverName}}: {{detail}}",
        [I18nKey.CONVERSATION$MCP_UNAVAILABLE_SETTINGS_LINK]:
          "<settingsLink>Check MCP settings</settingsLink>.",
        [I18nKey.BUTTON$CLOSE]: "Close",
        [I18nKey.SETTINGS$MCP_HEALTH_UNHEALTHY]: "Unhealthy",
      },
    },
  },
  interpolation: { escapeValue: false },
});

const CONV = "conv-1";

const renderBanner = () =>
  render(
    <I18nextProvider i18n={i18n}>
      <MemoryRouter>
        <McpUnavailableBanner conversationId={CONV} />
      </MemoryRouter>
    </I18nextProvider>,
  );

const mockUnhealthy = (
  overrides: Partial<ReturnType<typeof useMcpConversationHealth>> = {},
) => {
  vi.mocked(useMcpConversationHealth).mockReturnValue({
    unhealthyServers: [
      {
        server: {
          id: "stdio-0",
          type: "stdio",
          name: "jira",
          command: "mcp-jira",
        },
        serverId: "jira",
        health: {
          server_id: "jira",
          status: "unhealthy",
          category: "connection",
          message: "Connection refused",
        },
      },
    ],
    isLoading: false,
    ...overrides,
  });
};

describe("<McpUnavailableBanner />", () => {
  beforeEach(() => {
    vi.mocked(useMcpConversationHealth).mockReturnValue({
      unhealthyServers: [],
      isLoading: false,
    });
    useMcpWarningDismissStore.setState({ dismissedKeys: [] });
  });

  it("renders nothing when there are no unhealthy servers", () => {
    const { container } = renderBanner();
    expect(container).toBeEmptyDOMElement();
  });

  it("renders a warning for an unhealthy MCP server", () => {
    mockUnhealthy();
    renderBanner();
    expect(screen.getByTestId("mcp-unavailable-banner")).toBeInTheDocument();
    expect(screen.getByTestId("mcp-unavailable-banner-content")).toHaveTextContent(
      "jira",
    );
    expect(screen.getByTestId("mcp-unavailable-banner-content")).toHaveTextContent(
      "Connection refused",
    );
    expect(screen.getByRole("link", { name: /check mcp settings/i })).toHaveAttribute(
      "href",
      "/settings/mcp",
    );
  });

  it("hides the banner after dismiss", async () => {
    mockUnhealthy();
    const user = userEvent.setup();
    renderBanner();
    await user.click(screen.getByTestId("mcp-unavailable-banner-dismiss"));
    expect(screen.queryByTestId("mcp-unavailable-banner")).not.toBeInTheDocument();
  });

  it("does not render while health is loading", () => {
    mockUnhealthy({ isLoading: true, unhealthyServers: [] });
    const { container } = renderBanner();
    expect(container).toBeEmptyDOMElement();
  });
});
