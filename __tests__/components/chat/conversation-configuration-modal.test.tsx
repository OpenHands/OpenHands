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

function savedServers() {
  server.use(
    http.get("*/api/settings", () =>
      HttpResponse.json({
        agent_settings: {
          mcp_config: {
            "my-fetch": {
              command: "python3",
              args: ["-m", "fetch"],
              enabled: true,
            },
            "my-time": {
              command: "python3",
              args: ["-m", "time"],
              enabled: true,
            },
            disabled: { url: "https://mcp.example.com", enabled: false },
          },
        },
      }),
    ),
  );
}

describe("ConversationConfigurationModal", () => {
  beforeEach(() => SettingsService.invalidateCache());

  it("grants a secret only to this conversation without persisting shared settings", async () => {
    const user = userEvent.setup();
    const requests: unknown[] = [];
    let settingsWrites = 0;
    server.use(
      http.post(
        "*/api/conversations/:id/configuration/changes",
        async ({ request, params }) => {
          expect(params.id).toBe(conversationId);
          requests.push(await request.json());
          return HttpResponse.json({
            applied: [{ type: "secret.grant", name: "MY_TOKEN" }],
          });
        },
      ),
      http.patch("*/api/settings", () => {
        settingsWrites += 1;
        return HttpResponse.json({});
      }),
      http.put("*/api/settings/secrets", () => {
        settingsWrites += 1;
        return HttpResponse.json({});
      }),
    );

    renderWithProviders(
      <ConversationConfigurationModal
        conversationId={conversationId}
        isAcp={false}
        onClose={() => {}}
      />,
    );
    await user.type(screen.getByTestId("conversation-secret-name"), "MY_TOKEN");
    await user.type(
      screen.getByTestId("conversation-secret-value"),
      " private value ",
    );
    await user.click(
      screen.getByRole("button", { name: I18nKey.SECRETS$ADD_SECRET }),
    );

    await waitFor(() =>
      expect(requests).toEqual([
        {
          operations: [
            {
              type: "secret.grant",
              name: "MY_TOKEN",
              value: " private value ",
            },
          ],
        },
      ]),
    );
    expect(settingsWrites).toBe(0);
    expect(screen.queryByText(" private value ")).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent(
      I18nKey.CHAT$CONVERSATION_CONFIGURATION_SUCCESS,
    );
  });

  it("attaches an enabled saved MCP ref, not its credentials or global settings", async () => {
    savedServers();
    const user = userEvent.setup();
    const requests: unknown[] = [];
    let settingsWrites = 0;
    server.use(
      http.post(
        "*/api/conversations/:id/configuration/changes",
        async ({ request }) => {
          requests.push(await request.json());
          return HttpResponse.json({
            applied: [{ type: "mcp.attach", ref: "my-fetch" }],
          });
        },
      ),
      http.patch("*/api/settings", () => {
        settingsWrites += 1;
        return HttpResponse.json({});
      }),
    );

    renderWithProviders(
      <ConversationConfigurationModal
        conversationId={conversationId}
        isAcp={false}
        onClose={() => {}}
      />,
    );
    const item = await screen.findByTestId("conversation-mcp-my-fetch");
    expect(
      screen.queryByTestId("conversation-mcp-disabled"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("conversation-mcp-add-selected")).toBeDisabled();
    await user.click(within(item).getByRole("checkbox"));
    expect(requests).toHaveLength(0);
    await user.click(screen.getByTestId("conversation-mcp-add-selected"));

    await waitFor(() =>
      expect(requests).toEqual([
        { operations: [{ type: "mcp.attach", ref: "my-fetch" }] },
      ]),
    );
    expect(settingsWrites).toBe(0);
    expect(screen.getByRole("status")).toHaveTextContent(
      I18nKey.CHAT$CONVERSATION_CONFIGURATION_SUCCESS,
    );
  });

  it("attaches checked MCPs individually and retries only a failed attachment", async () => {
    savedServers();
    const user = userEvent.setup();
    const refs: string[] = [];
    server.use(
      http.post(
        "*/api/conversations/:id/configuration/changes",
        async ({ request }) => {
          const body = (await request.json()) as {
            operations: { type: string; ref: string }[];
          };
          expect(body.operations).toHaveLength(1);
          refs.push(body.operations[0].ref);
          if (body.operations[0].ref === "my-time" && refs.length === 2) {
            return HttpResponse.json({ detail: "Try again" }, { status: 409 });
          }
          return HttpResponse.json({ applied: body.operations });
        },
      ),
    );

    renderWithProviders(
      <ConversationConfigurationModal
        conversationId={conversationId}
        isAcp={false}
        onClose={() => {}}
      />,
    );
    await user.click(
      within(await screen.findByTestId("conversation-mcp-my-fetch")).getByRole(
        "checkbox",
      ),
    );
    await user.click(
      within(screen.getByTestId("conversation-mcp-my-time")).getByRole(
        "checkbox",
      ),
    );
    expect(refs).toEqual([]);
    await user.click(screen.getByTestId("conversation-mcp-add-selected"));
    await waitFor(() => expect(refs).toEqual(["my-fetch", "my-time"]));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      I18nKey.CHAT$CONVERSATION_CONFIGURATION_FAILED,
    );
    expect(
      within(screen.getByTestId("conversation-mcp-my-fetch")).getByRole(
        "checkbox",
      ),
    ).toBeDisabled();
    await user.click(screen.getByTestId("conversation-mcp-add-selected"));
    await waitFor(() =>
      expect(refs).toEqual(["my-fetch", "my-time", "my-time"]),
    );
    expect(screen.getByTestId("conversation-mcp-add-selected")).toBeDisabled();
  });

  it("installs a marketplace server only into the current conversation without writing settings", async () => {
    savedServers();
    const user = userEvent.setup();
    const requests: unknown[] = [];
    let settingsWrites = 0;
    server.use(
      http.post("*/api/mcp/test", () =>
        HttpResponse.json({ ok: true, tools: ["slack_post_message"] }),
      ),
      http.post(
        "*/api/conversations/:id/configuration/changes",
        async ({ request }) => {
          requests.push(await request.json());
          return HttpResponse.json({
            applied: [{ type: "mcp.install", name: "slack" }],
          });
        },
      ),
      http.patch("*/api/settings", () => {
        settingsWrites += 1;
        return HttpResponse.json({});
      }),
      http.put("*/api/settings/secrets", () => {
        settingsWrites += 1;
        return HttpResponse.json({});
      }),
    );

    renderWithProviders(
      <ConversationConfigurationModal
        conversationId={conversationId}
        isAcp={false}
        canInstallMcp
        onClose={() => {}}
      />,
    );
    await user.click(
      within(await screen.findByTestId("mcp-marketplace-card-slack")).getByRole(
        "checkbox",
      ),
    );
    expect(requests).toHaveLength(0);
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

    await waitFor(() => expect(requests).toHaveLength(1));
    expect(requests).toEqual([
      {
        operations: [
          {
            type: "mcp.install",
            name: "slack",
            server: expect.objectContaining({
              transport: "stdio",
              env: { SLACK_BOT_TOKEN: "scoped-token", SLACK_TEAM_ID: "T02" },
            }),
          },
        ],
      },
    ]);
    expect(settingsWrites).toBe(0);
    expect(await screen.findByRole("status")).toHaveTextContent(
      I18nKey.CHAT$CONVERSATION_CONFIGURATION_SUCCESS,
    );
  });

  it("installs a custom MCP server only into this conversation", async () => {
    const user = userEvent.setup();
    const requests: unknown[] = [];
    let settingsWrites = 0;
    server.use(
      http.post(
        "*/api/conversations/:id/configuration/changes",
        async ({ request }) => {
          requests.push(await request.json());
          return HttpResponse.json({
            applied: [{ type: "mcp.install", name: "ephemeral" }],
          });
        },
      ),
      http.patch("*/api/settings", () => {
        settingsWrites += 1;
        return HttpResponse.json({});
      }),
    );

    renderWithProviders(
      <ConversationConfigurationModal
        conversationId={conversationId}
        isAcp={false}
        canInstallMcp
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

    await waitFor(() => expect(requests).toHaveLength(1));
    expect(requests).toEqual([
      {
        operations: [
          {
            type: "mcp.install",
            name: "ephemeral",
            server: { transport: "sse", url: "https://mcp.example.com/api" },
          },
        ],
      },
    ]);
    expect(settingsWrites).toBe(0);
  });

  it("installs multiple selected marketplace MCPs with one operation per request", async () => {
    const user = userEvent.setup();
    const requests: { operations: { type: string; name: string }[] }[] = [];
    server.use(
      http.post("*/api/mcp/test", () =>
        HttpResponse.json({ ok: true, tools: [] }),
      ),
      http.post(
        "*/api/conversations/:id/configuration/changes",
        async ({ request, params }) => {
          expect(params.id).toBe(conversationId);
          const body = (await request.json()) as (typeof requests)[number];
          requests.push(body);
          return HttpResponse.json({ applied: body.operations });
        },
      ),
    );
    renderWithProviders(
      <ConversationConfigurationModal
        conversationId={conversationId}
        isAcp={false}
        canInstallMcp
        onClose={() => {}}
      />,
    );
    await user.click(
      within(await screen.findByTestId("mcp-marketplace-card-fetch")).getByRole(
        "checkbox",
      ),
    );
    await user.click(
      within(screen.getByTestId("mcp-marketplace-card-time")).getByRole(
        "checkbox",
      ),
    );
    expect(requests).toEqual([]);
    await user.click(screen.getByTestId("conversation-mcp-install-selected"));
    await user.click(await screen.findByTestId("mcp-install-submit"));
    await waitFor(() => expect(requests).toHaveLength(1));
    await waitFor(() =>
      expect(screen.getByTestId("mcp-install-modal")).toHaveTextContent("Time"),
    );
    await user.click(screen.getByTestId("mcp-install-submit"));
    await waitFor(() => expect(requests).toHaveLength(2));
    expect(requests.map(({ operations }) => operations)).toMatchObject([
      [{ type: "mcp.install", name: "fetch" }],
      [{ type: "mcp.install", name: "time" }],
    ]);
    expect(
      await screen.findByTestId("conversation-mcp-install-selected"),
    ).toBeDisabled();
  });

  it("keeps secret entry after an access error and permits retry without displaying the value", async () => {
    const user = userEvent.setup();
    let attempts = 0;
    server.use(
      http.post("*/api/conversations/:id/configuration/changes", () => {
        attempts += 1;
        return attempts === 1
          ? HttpResponse.json({ detail: "Secret not allowed" }, { status: 403 })
          : HttpResponse.json({
              applied: [{ type: "secret.grant", name: "MY_TOKEN" }],
            });
      }),
    );

    renderWithProviders(
      <ConversationConfigurationModal
        conversationId={conversationId}
        isAcp
        onClose={() => {}}
      />,
    );
    expect(
      screen.queryByTestId("conversation-mcp-section"),
    ).not.toBeInTheDocument();
    await user.type(screen.getByTestId("conversation-secret-name"), "MY_TOKEN");
    await user.type(
      screen.getByTestId("conversation-secret-value"),
      "unsafe-value",
    );
    await user.click(
      screen.getByRole("button", { name: I18nKey.SECRETS$ADD_SECRET }),
    );
    await screen.findByRole("alert");
    expect(screen.getByTestId("conversation-secret-value")).toHaveValue(
      "unsafe-value",
    );
    expect(screen.queryByText("unsafe-value")).not.toBeInTheDocument();
    await user.click(
      screen.getByRole("button", { name: I18nKey.SECRETS$ADD_SECRET }),
    );
    await waitFor(() => expect(attempts).toBe(2));
    expect(await screen.findByRole("status")).toHaveTextContent(
      I18nKey.CHAT$CONVERSATION_CONFIGURATION_SUCCESS,
    );
  });
});
