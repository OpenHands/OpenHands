import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { http, HttpResponse, delay } from "msw";
import { server } from "#/mocks/node";
import { useBashCommandRunner } from "#/hooks/use-bash-command-runner";
import { supportsConversationRuntimeRoutes } from "#/api/agent-server-client-options";
import {
  setRegisteredBackends,
  setActiveSelection,
} from "#/api/backend-registry/active-store";

const host = "http://localhost:9876";
const cid = "11111111-1111-4111-8111-111111111111";

beforeEach(() => {
  setRegisteredBackends([
    {
      id: "bash-probe",
      name: "Probe",
      kind: "local",
      host,
      apiKey: "backend-key",
    },
  ]);
  setActiveSelection({ backendId: "bash-probe" });
  server.use(
    http.get(`${host}/server_info`, () =>
      HttpResponse.json({
        version: "1.47.0",
        capabilities: ["conversation_runtime_routes_v1"],
        conversation_runtime: "docker",
      }),
    ),
  );
});

describe("useBashCommandRunner", () => {
  it.skipIf(!supportsConversationRuntimeRoutes())(
    "scopes commands before URL hydration and correlates concurrent results",
    async () => {
      const seen: string[] = [];
      server.use(
        http.post(
          `${host}/api/conversations/${cid}/bash/execute_bash_command`,
          async ({ request }) => {
            expect(request.headers.get("X-Session-API-Key")).toBe(
              "conversation-key",
            );
            const body = (await request.json()) as {
              command: string;
              cwd: string;
              timeout: number;
            };
            expect(body.cwd).toBe("/workspace");
            expect(body.timeout).toBe(30);
            seen.push(body.command);
            if (body.command === "first") await delay(20);
            return HttpResponse.json({
              exit_code: 0,
              stdout: body.command,
              stderr: "",
            });
          },
        ),
      );
      const { result } = renderHook(() =>
        useBashCommandRunner(undefined, "conversation-key", true, cid),
      );
      const output = await Promise.all([
        result.current("first", "/workspace", 30),
        result.current("second", "/workspace", 30),
      ]);
      expect(output.map((x) => x.stdout)).toEqual(["first", "second"]);
      expect(seen).toEqual(["first", "second"]);
    },
  );

  it("rejects a disabled probe without sending a request", async () => {
    const { result } = renderHook(() =>
      useBashCommandRunner(undefined, null, false, cid),
    );
    await expect(result.current("pwd", "/workspace", 30)).rejects.toThrow(
      "disabled",
    );
  });

  it.skipIf(!supportsConversationRuntimeRoutes())(
    "propagates runtime failures without claiming command success",
    async () => {
      server.use(
        http.post(
          `${host}/api/conversations/${cid}/bash/execute_bash_command`,
          () =>
            HttpResponse.json(
              { detail: "Runtime unavailable" },
              { status: 503 },
            ),
        ),
      );
      const { result } = renderHook(() =>
        useBashCommandRunner(`${host}/api/conversations/${cid}`, null, true),
      );
      await expect(result.current("pwd", "/workspace", 30)).rejects.toThrow();
    },
  );
});
