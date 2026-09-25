/** Test-scoped ownership for the six atomic mock-LLM journeys (#15481).
 * The runner still uses one worker: the backend and mock trajectory are shared.
 * A new test owns its prerequisites and teardown, never a predecessor's state.
 */
import { randomUUID } from "node:crypto";
import { test as base, expect, type Response } from "@playwright/test";
import {
  AgentProfilesClient,
  ProfilesClient,
} from "@openhands/typescript-client/clients";
import {
  BACKEND_URL,
  SESSION_API_KEY,
  deleteConversation,
  resetMockLLM,
  seedLocalStorage,
} from "./mock-llm-helpers";

export interface AtomicJourney {
  profileName: string;
  newProfileName: (label: string) => string;
  conversationIds: Set<string>;
  cleanup: (
    name: string,
    action: () => void | Promise<void>,
    phase?: "before-conversations" | "after-conversations",
  ) => void;
}

export const test = base.extend<{ journey: AtomicJourney }>({
  journey: [
    async ({ page, request }, runTest, testInfo) => {
      const options = { host: BACKEND_URL, apiKey: SESSION_API_KEY };
      const agents = new AgentProfilesClient(options);
      const profiles = new ProfilesClient(options);
      // ensureMockLLMProfile and the ACP editor change the seeded default
      // agent. Restore it before deleting the LLM profiles it referenced.
      // The runner probes automation readiness; the Agent Server may still
      // be booting independently on a cold uvx cache.
      await expect
        .poll(
          async () => {
            const response = await request.get(`${BACKEND_URL}/server_info`, {
              headers: { "X-Session-API-Key": SESSION_API_KEY },
            });
            return response.status();
          },
          { timeout: 60_000, intervals: [500, 1_000, 2_000] },
        )
        .toBe(200);
      const originalAgents = await agents.listAgentProfiles();
      const originalDefault = await agents.getAgentProfile("default");
      const originalProfiles = await profiles.listProfiles();
      const profileNames = new Set<string>();
      const conversationIds = new Set<string>();
      const cleanups: Array<{
        name: string;
        action: () => void | Promise<void>;
        phase: "before-conversations" | "after-conversations";
      }> = [];
      const pendingResponses = new Set<Promise<void>>();
      const errors: Error[] = [];
      const prefix = `atomic-${randomUUID().slice(0, 8)}`;
      const newProfileName = (label: string) => {
        const name = `${prefix}-${label}`;
        profileNames.add(name);
        return name;
      };
      // Capture successful creates even when navigation/assertions fail before
      // the test can obtain the ID from the URL.
      const captureConversation = (response: Response) => {
        if (
          response.request().method() !== "POST" ||
          new URL(response.url()).origin !== new URL(BACKEND_URL).origin ||
          new URL(response.url()).pathname !== "/api/conversations" ||
          !response.ok()
        )
          return;
        const pending = response
          .json()
          .then((body) => {
            const id = body.id ?? body.conversation_id;
            if (typeof id === "string") conversationIds.add(id);
          })
          .catch((cause: unknown) => {
            errors.push(
              new Error("Read created conversation for cleanup", { cause }),
            );
          });
        pendingResponses.add(pending);
        void pending.finally(() => pendingResponses.delete(pending));
      };
      page.on("response", captureConversation);
      const attempt = async (
        name: string,
        action: () => void | Promise<void>,
      ) => {
        try {
          await action();
        } catch (cause) {
          errors.push(new Error(`Journey cleanup failed: ${name}`, { cause }));
        }
      };
      try {
        await seedLocalStorage(page);
        await resetMockLLM(request);
        await runTest({
          profileName: newProfileName("llm"),
          newProfileName,
          conversationIds,
          cleanup: (name, action, phase = "after-conversations") =>
            cleanups.push({ name, action, phase }),
        });
      } catch (cause) {
        errors.push(new Error("Journey setup or execution failed", { cause }));
      } finally {
        if (testInfo.status !== testInfo.expectedStatus && !page.isClosed()) {
          // Preserve the failing UI before navigation stops reconciliation.
          try {
            await testInfo.attach("before-journey-teardown", {
              body: await page.screenshot({
                mask: [page.locator('input[type="password"]')],
              }),
              contentType: "image/png",
            });
          } catch {
            // Diagnostics must not prevent resource cleanup after a failure.
          }
        }
        // Stop browser effects from reactivating profiles during teardown.
        await attempt("stop page", async () => {
          await page.goto("about:blank");
        });
        page.off("response", captureConversation);
        await Promise.all(pendingResponses);
        // Automation cleanup also discovers run-created conversations; execute
        // it before deleting all the test's conversations.
        for (const { name, action, phase } of [...cleanups].reverse())
          if (phase === "before-conversations") await attempt(name, action);
        for (const id of conversationIds) {
          await attempt(`conversation ${id}`, () =>
            deleteConversation(request, id),
          );
        }
        for (const { name, action, phase } of [...cleanups].reverse())
          if (phase === "after-conversations") await attempt(name, action);
        await attempt("restore default agent", async () => {
          await agents.saveAgentProfile("default", originalDefault.profile);
          if (originalAgents.active_agent_profile_id) {
            await agents.activateAgentProfile(
              originalAgents.active_agent_profile_id,
            );
          }
        });
        for (const name of profileNames) {
          await attempt(`LLM profile ${name}`, async () => {
            try {
              await profiles.deleteProfile(name);
            } catch (error) {
              // Setup may fail before this reserved name is actually saved.
              if (
                !(error instanceof Error) ||
                !("status" in error) ||
                error.status !== 404
              )
                throw error;
            }
          });
        }
        if (originalProfiles.active_profile) {
          await attempt("restore active LLM profile", async () => {
            await profiles.activateProfile(originalProfiles.active_profile!);
          });
        }
        await attempt("reset mock LLM", () => resetMockLLM(request));
        agents.close();
        profiles.close();
      }
      if (errors.length)
        throw new AggregateError(
          errors,
          errors.map((error) => error.message).join("; "),
        );
    },
    { auto: true, timeout: 90_000 },
  ],
});
