import { expect, test } from "@playwright/test";

const CLOUD_HOST = "https://cloud.example.test";
const CONVERSATION_ID = "cloud-conversation-1";

test("renames a cloud conversation and keeps the title after refetch", async ({
  page,
}) => {
  test.setTimeout(60_000);

  await page.addInitScript(
    ({ cloudHost, conversationId }) => {
      const originalFetch = window.fetch.bind(window);
      const jsonResponse = (body: unknown) =>
        new Response(JSON.stringify(body), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      const getTitle = () =>
        window.localStorage.getItem("cloud-e2e-title") ??
        "Original cloud title";
      const getConversation = () => ({
        id: conversationId,
        created_by_user_id: "user-1",
        selected_repository: null,
        selected_branch: null,
        git_provider: null,
        title: getTitle(),
        trigger: null,
        pr_number: [],
        llm_model: null,
        metrics: null,
        created_at: "2026-01-01T00:00:00.000Z",
        updated_at: "2026-01-01T00:00:00.000Z",
        execution_status: "idle",
        conversation_url: null,
        session_api_key: null,
        sandbox_id: null,
        workspace: { working_dir: "/workspace/project" },
        public: false,
        sub_conversation_ids: [],
      });

      const state = window as unknown as {
        __cloudRenameRequests: Array<{
          authorization: string | null;
          body: unknown;
        }>;
      };
      state.__cloudRenameRequests = [];

      window.fetch = async (input, init) => {
        const request = new Request(input, init);
        const url = new URL(request.url);

        if (url.origin !== cloudHost) {
          return originalFetch(input, init);
        }

        if (url.pathname === "/api/keys/current") {
          return jsonResponse({ org_id: null });
        }

        if (url.pathname === "/api/v1/settings") {
          return jsonResponse({
            llm_model: "openhands/claude-haiku-4-5-20251001",
            llm_api_key_set: true,
            agent_settings: { agent_kind: "openhands" },
            user_consents_to_analytics: false,
          });
        }

        if (url.pathname === "/api/v1/settings/profiles") {
          return jsonResponse({ profiles: [], active_profile: null });
        }

        if (url.pathname === "/api/agent-profiles") {
          return jsonResponse({
            profiles: [],
            active_agent_profile_id: null,
          });
        }

        if (url.pathname === "/api/organizations") {
          return jsonResponse({ items: [], current_org_id: null });
        }

        if (url.pathname === "/api/v1/skills/search") {
          return jsonResponse({ items: [], next_page_id: null });
        }

        if (url.pathname === "/api/automation/sdk-version") {
          return jsonResponse({ version: null });
        }

        if (
          request.method === "GET" &&
          url.pathname === "/api/v1/app-conversations/search"
        ) {
          return jsonResponse({
            items: [getConversation()],
            next_page_id: null,
          });
        }

        if (
          request.method === "PATCH" &&
          url.pathname === `/api/v1/app-conversations/${conversationId}`
        ) {
          const body = (await request.clone().json()) as { title: string };
          window.localStorage.setItem("cloud-e2e-title", body.title);
          state.__cloudRenameRequests.push({
            authorization: request.headers.get("authorization"),
            body,
          });
          return jsonResponse(getConversation());
        }

        return jsonResponse({});
      };

      window.localStorage.setItem("analytics-consent", "false");
      window.localStorage.setItem("openhands-telemetry-consent", "denied");
      window.localStorage.setItem("openhands-telemetry-first-use", "true");
      window.localStorage.setItem("openhands-onboarded", "1");
      window.localStorage.setItem(
        "openhands-backends",
        JSON.stringify([
          {
            id: "cloud-e2e",
            name: "Cloud E2E",
            host: cloudHost,
            apiKey: "cloud-test-token",
            kind: "cloud",
          },
        ]),
      );
      window.localStorage.setItem(
        "openhands-active-backend",
        JSON.stringify({ backendId: "cloud-e2e", orgId: null }),
      );
    },
    { cloudHost: CLOUD_HOST, conversationId: CONVERSATION_ID },
  );

  await page.goto("/conversations");

  const originalCard = page
    .getByTestId("conversation-card")
    .filter({ hasText: "Original cloud title" });
  await expect(originalCard).toBeVisible({ timeout: 20_000 });
  const card = page.getByTestId("conversation-card").first();
  await card.hover();
  await card.getByTestId("ellipsis-button").click();
  await page.getByTestId("edit-button").click();

  const titleInput = card.getByTestId("conversation-card-title");
  await titleInput.fill("Renamed in Cloud");
  await titleInput.press("Enter");

  await expect
    .poll(() =>
      page.evaluate(
        () =>
          (
            window as unknown as {
              __cloudRenameRequests: Array<{
                authorization: string | null;
                body: unknown;
              }>;
            }
          ).__cloudRenameRequests,
      ),
    )
    .toEqual([
      {
        authorization: "Bearer cloud-test-token",
        body: { title: "Renamed in Cloud" },
      },
    ]);
  await expect(card.getByTestId("conversation-card-title")).toHaveText(
    "Renamed in Cloud",
  );

  await page.reload();
  await expect(
    page
      .getByTestId("conversation-card")
      .filter({ hasText: "Renamed in Cloud" }),
  ).toBeVisible({ timeout: 20_000 });
});
