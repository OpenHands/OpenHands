import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { server } from "#/mocks/node";
import { SecretRequestCard } from "#/components/features/conversation/secret-request-card";
import { handleRequestSecretAction } from "#/services/secret-request";
import { useSecretRequestStore } from "#/stores/secret-request-store";
import { useGoalStore } from "#/stores/goal-store";
import {
  REQUEST_SECRET_ACTION_KIND,
  SECRET_REQUEST_RESULT_PREFIX,
} from "#/constants/secret-request";
import { renderWithProviders } from "../../../../test-utils";

const name = "ROTATION_TEST_KEY";
const conversationId = "test-conversation-id";
const oldValue = "old-test-value";
let storedValue: string | undefined;
let writes: unknown[];
let attachments: unknown[];
let messages: string[];
let valueReads: number;

function PendingCards() {
  const requests = useSecretRequestStore((s) => s.requests);
  return requests.map((request) => (
    <SecretRequestCard key={request.toolCallId} request={request} />
  ));
}

beforeEach(() => {
  window.localStorage.clear();
  useSecretRequestStore.setState({ requests: [] });
  useGoalStore.setState({ statusByConversation: {} });
  storedValue = oldValue;
  writes = [];
  attachments = [];
  messages = [];
  valueReads = 0;
  // Only the HTTP server boundary is simulated: the form, mutations, services,
  // request queue, and typed clients all run their production code.
  server.use(
    http.get("*/api/settings/secrets", () =>
      HttpResponse.json({
        secrets: storedValue === undefined ? [] : [{ name }],
      }),
    ),
    http.get("*/api/settings/secrets/:name", () => {
      valueReads += 1;
      return HttpResponse.text(storedValue ?? "");
    }),
    http.put("*/api/settings/secrets", async ({ request }) => {
      const body = (await request.json()) as { name: string; value: string };
      writes.push(body);
      storedValue = body.value;
      return HttpResponse.json({ name: body.name });
    }),
    http.get("*/api/conversations", () =>
      HttpResponse.json([
        {
          id: conversationId,
          execution_status: "idle",
          conversation_url: null,
        },
      ]),
    ),
    http.post("*/api/conversations/:id/secrets", async ({ request }) => {
      attachments.push(await request.json());
      return HttpResponse.json({ success: true });
    }),
    http.post("*/api/conversations/:id/events", async ({ request }) => {
      const body = (await request.json()) as { content: { text: string }[] };
      messages.push(body.content[0].text);
      return HttpResponse.json({ ok: true });
    }),
  );
});

async function requestSecret() {
  const action = {
    kind: REQUEST_SECRET_ACTION_KIND,
    name,
    reason: "Rotate the credential",
  } as const;
  await handleRequestSecretAction(action, conversationId, "rotation-call");
  await handleRequestSecretAction(action, conversationId, "rotation-call");
  expect(useSecretRequestStore.getState().requests).toHaveLength(1);
  expect(messages).toEqual([]);
  expect(attachments).toEqual([]);
  renderWithProviders(<PendingCards />);
  const input = await screen.findByTestId("value-input");
  expect(input).toHaveValue("");
  expect(input).toHaveAttribute("type", "password");
  expect(screen.getByTestId("submit-button")).toBeDisabled();
  return input;
}

describe("secret request user flow", () => {
  it("does not overwrite another saved name when the user renames the request", async () => {
    const user = userEvent.setup();
    server.use(
      http.get("*/api/settings/secrets", () =>
        HttpResponse.json({ secrets: [{ name }, { name: "OTHER_KEY" }] }),
      ),
    );
    await requestSecret();
    await user.clear(screen.getByTestId("name-input"));
    await user.type(screen.getByTestId("name-input"), "OTHER_KEY");
    await user.type(screen.getByTestId("value-input"), "do-not-save");
    await user.click(screen.getByTestId("submit-button"));

    expect(
      await screen.findByText("SECRETS$SECRET_ALREADY_EXISTS"),
    ).toBeInTheDocument();
    expect(storedValue).toBe(oldValue);
    expect(writes).toEqual([]);
    expect(messages).toEqual([]);
  });

  it.each([true, false])(
    "saves only after user entry (existing secret: %s)",
    async (existing) => {
      if (!existing) storedValue = undefined;
      const user = userEvent.setup();
      const input = await requestSecret();
      await user.type(input, "new-test-value");
      await user.click(screen.getByTestId("submit-button"));

      await waitFor(() => expect(messages).toHaveLength(1));
      expect(storedValue).toBe("new-test-value");
      expect(writes).toEqual([{ name, value: "new-test-value" }]);
      expect(attachments).toEqual([
        {
          secrets: {
            [name]: expect.objectContaining({
              kind: "LookupSecret",
              url: `/api/settings/secrets/${name}`,
            }),
          },
        },
      ]);
      expect(messages).toEqual([
        `${SECRET_REQUEST_RESULT_PREFIX}${JSON.stringify({ status: "saved", name })}`,
      ]);
      expect(valueReads).toBe(0);
      expect(JSON.stringify(attachments)).not.toContain("new-test-value");
      expect(
        screen.queryByTestId("secret-request-card"),
      ).not.toBeInTheDocument();
    },
  );

  it.each([true, false])(
    "cancels without changing storage (existing secret: %s)",
    async (existing) => {
      if (!existing) storedValue = undefined;
      const user = userEvent.setup();
      await requestSecret();
      await user.click(screen.getByTestId("cancel-button"));

      await waitFor(() => expect(messages).toHaveLength(1));
      expect(
        JSON.parse(messages[0].slice(SECRET_REQUEST_RESULT_PREFIX.length)),
      ).toMatchObject({ status: "not_saved", name });
      expect(storedValue).toBe(existing ? oldValue : undefined);
      expect(writes).toEqual([]);
      expect(attachments).toEqual([]);
      expect(valueReads).toBe(0);
    },
  );
});

describe("secret request user flow on a cloud backend", () => {
  const runtimeUrl = "https://runtime.example.dev";
  let attachStatus: string;

  beforeEach(() => {
    __resetActiveStoreForTests();
    setRegisteredBackends([
      {
        id: "cloud-1",
        name: "Cloud",
        host: "https://app.example.dev",
        apiKey: "key",
        kind: "cloud",
      },
    ]);
    setActiveSelection({ backendId: "cloud-1", orgId: null });
    attachStatus = "attached";
    server.use(
      http.get("*/api/v1/secrets/search", () =>
        HttpResponse.json({ items: [], next_page_id: null }),
      ),
      http.post("*/api/v1/secrets", async ({ request }) => {
        writes.push(await request.json());
        return HttpResponse.json({ id: name });
      }),
      http.post(
        "*/api/v1/app-conversations/:id/secrets",
        async ({ request, params }) => {
          attachments.push({ id: params.id, body: await request.json() });
          return HttpResponse.json({ status: attachStatus, message: null });
        },
      ),
      http.get("*/api/v1/app-conversations", () =>
        HttpResponse.json([
          {
            id: conversationId,
            conversation_url: `${runtimeUrl}/api/conversations/${conversationId}`,
            session_api_key: "session-key",
          },
        ]),
      ),
    );
  });

  afterEach(() => __resetActiveStoreForTests());

  async function saveRequestedSecret() {
    const user = userEvent.setup();
    await handleRequestSecretAction(
      { kind: REQUEST_SECRET_ACTION_KIND, name, reason: "Deploy" },
      conversationId,
      "cloud-call",
    );
    renderWithProviders(<PendingCards />);
    await user.type(await screen.findByTestId("value-input"), "cloud-value");
    await user.click(screen.getByTestId("submit-button"));
    await waitFor(() => expect(messages).toHaveLength(1));
    return JSON.parse(messages[0].slice(SECRET_REQUEST_RESULT_PREFIX.length));
  }

  it("attaches the saved secret by name and reports it saved", async () => {
    const result = await saveRequestedSecret();

    expect(writes).toEqual([{ name, value: "cloud-value" }]);
    expect(attachments).toEqual([
      { id: conversationId, body: { secret_name: name } },
    ]);
    expect(result).toEqual({ status: "saved", name });
    expect(JSON.stringify(attachments)).not.toContain("cloud-value");
    expect(messages[0]).not.toContain("cloud-value");
  });

  it.each([
    ["deferred", "not_saved"],
    ["not_found", "error"],
    ["error", "error"],
  ])(
    "never reports saved when the instance answers %s",
    async (status, reported) => {
      attachStatus = status;

      const result = await saveRequestedSecret();

      expect(attachments).toHaveLength(1);
      expect(result).toMatchObject({ status: reported, name });
    },
  );
});
