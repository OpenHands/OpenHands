import { execFileSync } from "node:child_process";
import { existsSync, rmSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { basename, resolve, sep } from "node:path";
import { shutdownPortLease } from "./allocate-mock-llm-ports";

/**
 * Global teardown for mock-LLM E2E tests.
 * Removes the isolated per-run state and fixture directory after all tests finish,
 * unless MOCK_LLM_PRESERVE_STATE is explicitly set.
 */
function isDeletableRunDir(resolved: string): boolean {
  const root = resolve("/");
  const home = resolve(homedir());
  const cwd = resolve(process.cwd());
  if (resolved === root || resolved === home || resolved === cwd) return false;
  if (
    cwd.startsWith(`${resolved}${sep}`) ||
    home.startsWith(`${resolved}${sep}`)
  ) {
    return false;
  }

  const inside = (parent: string) => resolved.startsWith(`${parent}${sep}`);
  if (inside(resolve(cwd, ".tmp")) || inside(resolve(tmpdir()))) return true;
  return basename(resolved).startsWith("mock-llm");
}

export default async function globalTeardown(): Promise<void> {
  await shutdownPortLease();

  const containerName = process.env.MOCK_LLM_CONTAINER_NAME;
  if (containerName && !process.env.MOCK_LLM_PRESERVE_STATE) {
    try {
      execFileSync("docker", ["rm", "-f", containerName], { stdio: "ignore" });
    } catch (err) {
      console.warn(`Failed to remove Docker container ${containerName}:`, err);
    }
  }

  const runDir = process.env.MOCK_LLM_RUN_DIR;
  if (runDir && !process.env.MOCK_LLM_PRESERVE_STATE && existsSync(runDir)) {
    const resolved = resolve(runDir);
    if (!isDeletableRunDir(resolved)) {
      console.warn(
        `Refusing to delete unsafe per-run state directory: ${resolved}`,
      );
      return;
    }

    try {
      rmSync(resolved, { recursive: true, force: true });
    } catch (err) {
      console.warn(`Failed to clean per-run state directory ${resolved}:`, err);
    }
  }
}
