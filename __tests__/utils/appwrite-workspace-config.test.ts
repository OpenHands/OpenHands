import { describe, expect, it } from "vitest";
import {
  buildAppwriteIntegrationsPatch,
  findWorkspaceIdForPath,
  getAppwriteConfigForWorkspace,
} from "#/utils/appwrite-workspace-config";

describe("appwrite-workspace-config", () => {
  it("reads per-workspace config and ignores legacy global shape", () => {
    expect(
      getAppwriteConfigForWorkspace(
        {
          appwrite: {
            enabled: true,
            endpoint: "https://legacy.example/v1",
            projectId: "legacy",
          } as never,
        },
        "ws-1",
      ).enabled,
    ).toBe(false);

    expect(
      getAppwriteConfigForWorkspace(
        {
          appwrite: {
            byWorkspace: {
              "ws-1": {
                enabled: true,
                endpoint: "https://cloud.appwrite.io/v1",
                projectId: "proj",
              },
            },
          },
        },
        "ws-1",
      ),
    ).toMatchObject({
      enabled: true,
      projectId: "proj",
    });
  });

  it("matches conversation paths to workspace ids", () => {
    const workspaces = [
      { id: "root", name: "Root", path: "/workspace/project" },
      { id: "nested", name: "Nested", path: "/workspace/project/app" },
    ];
    expect(findWorkspaceIdForPath(workspaces, "/workspace/project")).toBe(
      "root",
    );
    expect(
      findWorkspaceIdForPath(workspaces, "/workspace/project/app/src"),
    ).toBe("nested");
    expect(findWorkspaceIdForPath(workspaces, "/other")).toBeNull();
  });

  it("patches one workspace without wiping siblings", () => {
    const patch = buildAppwriteIntegrationsPatch(
      {
        appwrite: {
          byWorkspace: {
            "ws-a": {
              enabled: true,
              endpoint: "https://a.example/v1",
              projectId: "a",
            },
          },
        },
      },
      "ws-b",
      {
        enabled: true,
        endpoint: "https://b.example/v1",
        projectId: "b",
      },
    );

    expect(Object.keys(patch.appwrite?.byWorkspace ?? {})).toEqual([
      "ws-a",
      "ws-b",
    ]);
    expect(patch.appwrite?.byWorkspace["ws-b"].projectId).toBe("b");
  });
});
