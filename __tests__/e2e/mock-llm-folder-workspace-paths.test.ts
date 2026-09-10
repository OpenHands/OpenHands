import path from "node:path";
import { describe, expect, it } from "vitest";

import {
  getFolderBrowserPathSegments,
  getFolderBrowserRootPath,
  resolveFolderWorkspacePaths,
  TEST_DIR_NAME,
  WORKSPACE_DIR_NAME,
} from "../../tests/e2e/mock-llm/utils/folder-workspace-paths";

describe("mock-LLM folder workspace paths", () => {
  it("keeps npm-mode Windows host and agent-server paths identical", () => {
    const tmpDir = String.raw`C:\Users\me\AppData\Local\Temp`;

    const paths = resolveFolderWorkspacePaths({
      tmpDir,
      env: {},
    });

    const expectedBase = path.win32.join(tmpDir, WORKSPACE_DIR_NAME);
    const expectedDir = path.win32.join(expectedBase, TEST_DIR_NAME);
    expect(paths.hostDirBase).toBe(expectedBase);
    expect(paths.containerDirBase).toBe(expectedBase);
    expect(paths.hostDir).toBe(expectedDir);
    expect(paths.testDir).toBe(expectedDir);
  });

  it("places each run below the fixed host and container roots", () => {
    const runDirName = "issue-16714-run-123";
    const paths = resolveFolderWorkspacePaths({
      env: {
        MOCK_LLM_FOLDER_WORKSPACE_HOST_DIR: "/host/folder-workspace",
        MOCK_LLM_FOLDER_WORKSPACE_CONTAINER_DIR: "/container/folder-workspace",
      },
      runDirName,
    });

    expect(paths.hostRunDir).toBe(`/host/folder-workspace/${runDirName}`);
    expect(paths.containerRunDir).toBe(
      `/container/folder-workspace/${runDirName}`,
    );
    expect(paths.hostDir).toBe(
      `/host/folder-workspace/${runDirName}/${TEST_DIR_NAME}`,
    );
    expect(paths.testDir).toBe(
      `/container/folder-workspace/${runDirName}/${TEST_DIR_NAME}`,
    );
  });

  it("derives Windows folder-browser roots and path segments", () => {
    const target = String.raw`C:\Users\me\AppData\Local\Temp\e2e-folder-workspace-test\my-test-project`;

    expect(getFolderBrowserRootPath(target)).toBe("C:\\");
    expect(getFolderBrowserPathSegments(target)).toEqual([
      "Users",
      "me",
      "AppData",
      "Local",
      "Temp",
      "e2e-folder-workspace-test",
      "my-test-project",
    ]);
  });
});
