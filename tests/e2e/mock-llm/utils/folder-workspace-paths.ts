import * as os from "node:os";
import * as path from "node:path";

export const WORKSPACE_DIR_NAME = "e2e-folder-workspace-test";
export const TEST_DIR_NAME = "my-test-project";

interface ResolveFolderWorkspacePathsOptions {
  env?: NodeJS.ProcessEnv;
  tmpDir?: string;
}

export function resolveFolderWorkspacePaths({
  env = process.env,
  tmpDir = os.tmpdir(),
}: ResolveFolderWorkspacePathsOptions = {}) {
  const hostDirBase =
    env.MOCK_LLM_FOLDER_WORKSPACE_HOST_DIR ??
    joinRuntimePath(tmpDir, WORKSPACE_DIR_NAME);
  const containerDirBase =
    env.MOCK_LLM_FOLDER_WORKSPACE_CONTAINER_DIR ?? hostDirBase;

  return {
    hostDirBase,
    containerDirBase,
    hostDir: joinRuntimePath(hostDirBase, TEST_DIR_NAME),
    testDir: joinRuntimePath(containerDirBase, TEST_DIR_NAME),
  };
}

export function getFolderBrowserRootPath(targetPath: string): string {
  const parser = getRuntimePathParser(targetPath);
  return parser.parse(targetPath).root || "/";
}

export function getFolderBrowserPathSegments(targetPath: string): string[] {
  const root = getFolderBrowserRootPath(targetPath);
  const relativePath = targetPath.slice(root.length);
  return relativePath.split(/[\\/]+/).filter(Boolean);
}

function joinRuntimePath(parent: string, child: string): string {
  return getRuntimePathParser(parent).join(parent, child);
}

function getRuntimePathParser(targetPath: string) {
  return /^[A-Za-z]:[\\/]/.test(targetPath) || targetPath.includes("\\")
    ? path.win32
    : path.posix;
}
