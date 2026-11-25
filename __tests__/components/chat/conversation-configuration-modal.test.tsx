import { http, HttpResponse } from "msw";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { renderWithProviders } from "test-utils";
import SettingsService from "#/api/settings-service/settings-service.api";
import { server } from "#/mocks/node";
import { I18nKey } from "#/i18n/declaration";
import { ConversationConfigurationModal } from "#/components/features/chat/conversation-configuration-modal";

const conversationId = "11111111-1111-4111-8111-111111111111";

describe("conversation additions", () => {
  beforeEach(() => SettingsService.invalidateCache());

  it("adds a secret only to the current conversation without revealing it", async () => {
    const requests: unknown[] = [];
    server.use(
      http.post(
        "*/api/conversations/:id/secrets",
        async ({ request, params }) => {
          expect(params.id).toBe(conversationId);
          requests.push(await request.json());
          return HttpResponse.json({ success: true });
        },
      ),
    );
    const user = userEvent.setup();
    renderWithProviders(
      <ConversationConfigurationModal
        conversationId={conversationId}
        section="secrets"
        onClose={() => {}}
      />,
    );

    await user.type(screen.getByTestId("conversation-secret-name"), "MY_TOKEN");
    await user.type(
      screen.getByTestId("conversation-secret-value"),
      "private-value",
    );
    await user.click(
      screen.getByRole("button", { name: I18nKey.SECRETS$ADD_SECRET }),
    );

    await waitFor(() =>
      expect(requests).toEqual([{ secrets: { MY_TOKEN: "private-value" } }]),
    );
    expect(screen.queryByText("private-value")).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("conversation-mcp-section"),
    ).not.toBeInTheDocument();
  });

  it("keeps the secret form retryable after a conversation-scoped failure", async () => {
    const requests: unknown[] = [];
    server.use(
      http.post("*/api/conversations/:id/secrets", async ({ request }) => {
        requests.push(await request.json());
        if (requests.length === 1) {
          return HttpResponse.json({ detail: "Forbidden" }, { status: 403 });
        }
        return HttpResponse.json({ success: true });
      }),
    );
    const user = userEvent.setup();
    renderWithProviders(
      <ConversationConfigurationModal
        conversationId={conversationId}
        section="secrets"
        onClose={() => {}}
      />,
    );

    await user.type(
      screen.getByTestId("conversation-secret-name"),
      "RETRY_TOKEN",
    );
    await user.type(
      screen.getByTestId("conversation-secret-value"),
      "retry-value",
    );
    const submit = screen.getByRole("button", {
      name: I18nKey.SECRETS$ADD_SECRET,
    });
    await user.click(submit);

    expect(await screen.findByRole("alert")).toBeInTheDocument();
    expect(screen.getByTestId("conversation-secret-name")).toHaveValue(
      "RETRY_TOKEN",
    );
    expect(screen.getByTestId("conversation-secret-value")).toHaveValue(
      "retry-value",
    );
    expect(submit).toBeEnabled();

    await user.click(submit);

    await waitFor(() => expect(requests).toHaveLength(2));
    expect(await screen.findByRole("status")).toBeInTheDocument();
    expect(requests).toEqual([
      { secrets: { RETRY_TOKEN: "retry-value" } },
      { secrets: { RETRY_TOKEN: "retry-value" } },
    ]);
  });

  it("attaches only enabled, supported saved MCPs without updating settings", async () => {
    server.use(
      http.get("*/api/settings", () =>
        HttpResponse.json({
          agent_settings: {
            mcp_config: {
              fetch: {
                command: "python3",
                args: ["-m", "fetch"],
                enabled: true,
              },
              legacy: {
                url: "https://example.com/sse",
                transport: "sse",
                enabled: true,
              },
              disabled: { command: "python3", enabled: false },
            },
          },
        }),
      ),
    );
    const requests: unknown[] = [];
    server.use(
      http.post("*/api/conversations/:id/mcp", async ({ request }) => {
        requests.push(await request.json());
        return HttpResponse.json({ success: true });
      }),
    );
    const user = userEvent.setup();
    renderWithProviders(
      <ConversationConfigurationModal
        conversationId={conversationId}
        section="mcp"
        onClose={() => {}}
      />,
    );

    const item = await screen.findByTestId("conversation-mcp-fetch");
    expect(
      screen.queryByTestId("conversation-mcp-legacy"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("conversation-mcp-disabled"),
    ).not.toBeInTheDocument();
    await user.click(within(item).getByRole("checkbox"));
    await user.click(screen.getByTestId("conversation-mcp-add-selected"));

    await waitFor(() =>
      expect(requests).toEqual([
        {
          mcp_servers: {
            fetch: {
              transport: "stdio",
              command: "python3",
              args: ["-m", "fetch"],
            },
          },
        },
      ]),
    );
    expect(
      screen.queryByTestId("conversation-secret-value"),
    ).not.toBeInTheDocument();
  });

  it("installs a marketplace template into this conversation without saving settings", async () => {
    const requests: unknown[] = [];
    server.use(
      http.post("*/api/mcp/test", () =>
        HttpResponse.json({ ok: true, tools: [] }),
      ),
      http.post("*/api/conversations/:id/mcp", async ({ request }) => {
        requests.push(await request.json());
        return HttpResponse.json({ success: true });
      }),
    );
    const user = userEvent.setup();
    renderWithProviders(
      <ConversationConfigurationModal
        conversationId={conversationId}
        section="mcp"
        onClose={() => {}}
      />,
    );

    await user.click(
      within(await screen.findByTestId("mcp-marketplace-card-slack")).getByRole(
        "checkbox",
      ),
    );
    await user.click(screen.getByTestId("conversation-mcp-install-selected"));
    await user.type(
      screen.getByTestId("mcp-install-field-SLACK_BOT_TOKEN"),
      "scoped-token",
    );
    await user.type(
      screen.getByTestId("mcp-install-field-SLACK_TEAM_ID"),
      "T02",
    );
    await user.click(screen.getByTestId("mcp-install-submit"));

    await waitFor(() =>
      expect(requests).toEqual([
        {
          mcp_servers: {
            slack: expect.objectContaining({
              transport: "stdio",
              env: { SLACK_BOT_TOKEN: "scoped-token", SLACK_TEAM_ID: "T02" },
            }),
          },
        },
      ]),
    );
    expect(screen.queryByText("scoped-token")).not.toBeInTheDocument();
  });

  it("defaults custom MCP additions to supported HTTP transport", async () => {
    const requests: unknown[] = [];
    server.use(
      http.post("*/api/conversations/:id/mcp", async ({ request }) => {
        requests.push(await request.json());
        return HttpResponse.json({ success: true });
      }),
    );
    const user = userEvent.setup();
    renderWithProviders(
      <ConversationConfigurationModal
        conversationId={conversationId}
        section="mcp"
        onClose={() => {}}
      />,
    );

    await user.click(
      screen.getByRole("button", { name: I18nKey.MCP$ADD_CUSTOM }),
    );
    await user.type(screen.getByTestId("server-name-input"), "ephemeral");
    await user.type(
      screen.getByTestId("url-input"),
      "https://mcp.example.com/api",
    );
    await user.click(screen.getByTestId("submit-button"));

    await waitFor(() =>
      expect(requests).toEqual([
        {
          mcp_servers: {
            ephemeral: {
              transport: "http",
              url: "https://mcp.example.com/api",
            },
          },
        },
      ]),
    );
  });

  it("saves a new LLM profile then routes the conversation through the model router", async () => {
    const actions: string[] = [];
    server.use(
      http.post("*/api/profiles/:name", async ({ request }) => {
        const body = (await request.json()) as { llm: { model: string } };
        actions.push(`save:${body.llm.model}`);
        return HttpResponse.json({ name: "test-model", message: "Saved" });
      }),
      http.post(
        "*/api/conversations/:id/model_router/routes",
        async ({ request }) => {
          const body = (await request.json()) as { profile_name: string };
          actions.push(`add-route:${body.profile_name}`);
          return HttpResponse.json({
            enabled: true,
            routes: [{ name: body.profile_name }],
          });
        },
      ),
      http.patch(
        "*/api/conversations/:id/model_router",
        async ({ request }) => {
          const body = (await request.json()) as {
            selection: { manual_route_name?: string };
          };
          actions.push(`switch-route:${body.selection.manual_route_name}`);
          return HttpResponse.json({
            enabled: true,
            selection: body.selection,
          });
        },
      ),
    );
    const user = userEvent.setup();
    renderWithProviders(
      <ConversationConfigurationModal
        conversationId={conversationId}
        section="models"
        onClose={() => {}}
      />,
    );

    await user.type(screen.getByTestId("conversation-llm-name"), "test-model");
    await user.type(
      screen.getByTestId("conversation-llm-model"),
      "openai/gpt-4o",
    );
    await user.click(
      screen.getByRole("button", { name: I18nKey.SETTINGS$ADD_LLM_PROFILE }),
    );

    await waitFor(() =>
      expect(actions).toEqual([
        "save:openai/gpt-4o",
        "add-route:test-model",
        "switch-route:test-model",
      ]),
    );
  });
});
