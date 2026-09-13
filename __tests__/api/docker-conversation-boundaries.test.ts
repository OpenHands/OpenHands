import { supportsConversationRuntimeRoutes } from "#/api/agent-server-client-options";
import AgentServerGitService from "#/api/git-service/agent-server-git-service.api";
import { ServerClient } from "@openhands/typescript-client/clients";
import { beforeEach, describe, expect, it } from "vitest";
import { http, HttpResponse } from "msw";
import { server } from "#/mocks/node";
import {
  setRegisteredBackends,
  setActiveSelection,
} from "#/api/backend-registry/active-store";
import { clearCachedAgentServerInfo } from "#/api/agent-server-compatibility";
import { clearAgentServerHomeDirCache } from "#/api/agent-server-home";
import {
  resolveNewConversationWorkspace,
  ISOLATED_WORKSPACE_MESSAGE,
} from "#/api/conversation-workspace";
import GitService from "#/api/git-service/git-service.api";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import AgentServerRuntimeService from "#/api/runtime-service/agent-server-runtime-service";

const host = "http://localhost:9876";
const cid = "11111111-1111-4111-8111-111111111111";
const paths: string[] = [];
let isolated = true;
let capabilities = true;

beforeEach(() => {
  isolated = true;
  capabilities = true;
  paths.length = 0;
  clearCachedAgentServerInfo();
  clearAgentServerHomeDirCache();
  setRegisteredBackends([
    {
      id: "boundary",
      name: "Boundary",
      kind: "local",
      host,
      apiKey: "boundary-test-key",
    },
  ]);
  setActiveSelection({ backendId: "boundary" });
  // Only the HTTP server is substituted; frontend services and SDK clients run unchanged.
  server.use(
    http.all(`${host}/*`, ({ request }) => {
      const url = new URL(request.url);
      paths.push(url.pathname + url.search);
      expect(request.headers.get("X-Session-API-Key")).toBe(
        "boundary-test-key",
      );
      if (url.pathname === "/server_info")
        return HttpResponse.json({
          version: "1.45.0",
          uptime: 0,
          idle_time: 0,
          capabilities: capabilities ? ["conversation_runtime_routes_v1"] : [],
          conversation_runtime: isolated ? "docker" : "local",
          workspace_mode: isolated ? "isolated" : "host",
          runtime_services: { mode: isolated ? "dev:automation" : "docker" },
        });
      if (url.pathname === "/api/file/home")
        return HttpResponse.json({ home: "/home/agent" });
      if (url.pathname === "/api/conversations")
        return HttpResponse.json([
          {
            id: cid,
            created_at: "2026-01-01T00:00:00Z",
            updated_at: "2026-01-01T00:00:00Z",
            execution_status: "idle",
            workspace: { working_dir: "/workspace" },
          },
        ]);
      if (url.pathname.endsWith("/git/changes"))
        return HttpResponse.json([{ path: "file.txt", status: "M" }]);
      if (url.pathname.endsWith("/git/diff"))
        return HttpResponse.json({ original: "before", modified: "after" });
      if (url.pathname.endsWith("/file/download"))
        return HttpResponse.text("plan contents");
      if (url.pathname.endsWith("/bash/execute_bash_command"))
        return HttpResponse.json({
          exit_code: 0,
          stdout: "/workspace",
          stderr: "",
        });
      return HttpResponse.json(
        { error: "Unexpected endpoint" },
        { status: 500 },
      );
    }),
  );
});

describe.skipIf(!supportsConversationRuntimeRoutes())(
  "conversation runtime boundaries",
  () => {
    it("scopes live unified git calls even before conversation URL hydration", async () => {
      expect(
        await AgentServerGitService.getGitChanges(
          cid,
          null,
          null,
          "/workspace",
        ),
      ).toHaveLength(1);
      expect(
        await AgentServerGitService.getGitChangeDiff(
          cid,
          null,
          null,
          "file.txt",
        ),
      ).toEqual({ original: "before", modified: "after" });
      expect(paths).toContain(
        `/api/conversations/${cid}/git/changes?path=%2Fworkspace`,
      );
      expect(paths).toContain(
        `/api/conversations/${cid}/git/diff?path=file.txt`,
      );
    });

    it("blocks isolated creation when the installed SDK cannot scope runtime calls", async () => {
      const descriptor = Object.getOwnPropertyDescriptor(
        ServerClient,
        "supportsConversationRuntimeRoutes",
      );
      Object.defineProperty(ServerClient, "supportsConversationRuntimeRoutes", {
        value: false,
        configurable: true,
      });
      try {
        await expect(
          resolveNewConversationWorkspace({ conversationId: cid }),
        ).rejects.toThrow("Upgrade Canvas");
      } finally {
        if (descriptor)
          Object.defineProperty(
            ServerClient,
            "supportsConversationRuntimeRoutes",
            descriptor,
          );
      }
    });

    it("uses a new isolated workspace without reading host home or hooks", async () => {
      expect(
        await resolveNewConversationWorkspace({ conversationId: cid }),
      ).toEqual({
        workingDir: "/workspace",
        hooksProjectDir: null,
        isolated: true,
      });
      expect(paths).toEqual(["/server_info"]);
    });

    it.each(["/home/user/project", "/workspace", "relative/project"])(
      "rejects selected host folder %s rather than discarding it",
      async (workingDir) => {
        await expect(
          resolveNewConversationWorkspace({ conversationId: cid, workingDir }),
        ).rejects.toThrow(ISOLATED_WORKSPACE_MESSAGE);
        expect(paths).toEqual(["/server_info"]);
      },
    );

    it("rejects selected host repository metadata", async () => {
      await expect(
        resolveNewConversationWorkspace({
          conversationId: cid,
          selectedRepository: "owner/project",
        }),
      ).rejects.toThrow(ISOLATED_WORKSPACE_MESSAGE);
    });

    it("preserves a Docker child conversation's parent workspace", async () => {
      expect(
        await resolveNewConversationWorkspace({
          conversationId: cid,
          parentConversationId: "parent",
          workingDir: "/workspace",
        }),
      ).toMatchObject({ workingDir: "/workspace", isolated: true });
    });

    it("uses host metadata rather than runtime_services deployment mode", async () => {
      isolated = false;
      expect(
        await resolveNewConversationWorkspace({
          conversationId: cid,
          workingDir: "/home/user/project",
        }),
      ).toEqual({
        workingDir: "/home/user/project",
        hooksProjectDir: "/home/user/project",
        isolated: false,
      });
    });

    it("routes legacy git service changes and diff to the conversation", async () => {
      expect(await GitService.getGitChanges(cid)).toHaveLength(1);
      expect(await GitService.getGitChangeDiff(cid, "file.txt")).toEqual({
        original: "before",
        modified: "after",
      });
      expect(paths).toContain(
        `/api/conversations/${cid}/git/changes?path=%2Fworkspace`,
      );
      expect(paths).toContain(
        `/api/conversations/${cid}/git/diff?path=file.txt`,
      );
    });

    it("routes plan downloads and runtime bash to the same conversation", async () => {
      expect(
        await AgentServerConversationService.readConversationFile(cid),
      ).toBe("plan contents");
      expect(
        await AgentServerRuntimeService.executeCommand(
          `${host}/api/conversations/${cid}`,
          null,
          "pwd",
          "/workspace",
        ),
      ).toMatchObject({ stdout: "/workspace" });
      expect(paths).toContain(
        `/api/conversations/${cid}/file/download?path=%2Fworkspace%2F.agents_tmp%2FPLAN.md`,
      );
      expect(paths).toContain(
        `/api/conversations/${cid}/bash/execute_bash_command`,
      );
    });

    it("retains cid for legacy servers without canonical capability", async () => {
      capabilities = false;
      await GitService.getGitChangeDiff(cid, "file.txt");
      expect(paths).toContain(`/api/git/diff?path=file.txt&cid=${cid}`);
    });
  },
);
