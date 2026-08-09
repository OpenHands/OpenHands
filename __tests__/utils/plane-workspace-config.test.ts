import { describe, expect, it } from "vitest";
import {
  buildPlaneIntegrationsPatch,
  getPlaneConfigForWorkspace,
} from "#/utils/plane-workspace-config";
import { planeApiKeySecretName } from "#/utils/plane-integration-secrets";

describe("plane-workspace-config", () => {
  it("reads per-workspace Plane config with empty default base URL", () => {
    expect(getPlaneConfigForWorkspace(undefined, "ws-1")).toMatchObject({
      enabled: false,
      baseUrl: "",
      workspaceSlug: "",
      projectId: "",
      moduleId: "",
    });

    expect(
      getPlaneConfigForWorkspace(
        {
          plane: {
            byWorkspace: {
              "ws-1": {
                enabled: true,
                baseUrl: "https://plane.example.com",
                workspaceSlug: "heimdall",
                projectId: "proj-1",
                moduleId: "mod-1",
              },
            },
          },
        },
        "ws-1",
      ),
    ).toMatchObject({
      enabled: true,
      baseUrl: "https://plane.example.com",
      workspaceSlug: "heimdall",
      projectId: "proj-1",
      moduleId: "mod-1",
    });
  });

  it("patches one workspace without wiping siblings or omitting secret name", () => {
    const patch = buildPlaneIntegrationsPatch(
      {
        plane: {
          byWorkspace: {
            "ws-a": {
              enabled: true,
              baseUrl: "https://a.example",
              workspaceSlug: "a",
              projectId: "a",
            },
          },
        },
      },
      "ws-b",
      {
        enabled: true,
        baseUrl: "https://b.example",
        workspaceSlug: "b",
        projectId: "b",
        moduleId: "",
      },
    );

    expect(Object.keys(patch.plane?.byWorkspace ?? {})).toEqual([
      "ws-a",
      "ws-b",
    ]);
    expect(patch.plane?.byWorkspace["ws-b"]).toMatchObject({
      projectId: "b",
      apiKeySecretName: planeApiKeySecretName("ws-b"),
    });
    expect(patch.plane?.byWorkspace["ws-b"].moduleId).toBeUndefined();
  });
});
