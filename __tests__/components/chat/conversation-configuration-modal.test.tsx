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
    await user.click(within(item).getByRole("button"));

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
