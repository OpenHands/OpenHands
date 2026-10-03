import { execSync } from "node:child_process";
import { existsSync, rmSync } from "node:fs";
import { homedir } from "node:os";
import { resolve } from "node:path";

/**
 * Global teardown for mock-LLM E2E tests.
 * Removes the isolated per-run state and fixture directory after all tests finish,
 * unless MOCK_LLM_PRESERVE_STATE is explicitly set.
 */
export default async function globalTeardown(): Promise<void> {
  const containerName = process.env.MOCK_LLM_CONTAINER_NAME;
  if (containerName && !process.env.MOCK_LLM_PRESERVE_STATE) {
    try {
      execSync(`docker rm -f ${containerName}`, { stdio: "ignore" });
    } catch {
      // best-effort container cleanup
    }
  }

  const runDir = process.env.MOCK_LLM_RUN_DIR;
  if (runDir && !process.env.MOCK_LLM_PRESERVE_STATE && existsSync(runDir)) {
    const resolved = resolve(runDir);
    const root = resolve("/");
    const home = resolve(homedir());
    const cwd = resolve(process.cwd());

    // Defensive check: never delete filesystem root, home directory, or repo root
    if (resolved === root || resolved === home || resolved === cwd) {
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
