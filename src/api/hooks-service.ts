import { HooksClient } from "@openhands/typescript-client/clients";
import type { HookConfig } from "@openhands/typescript-client";
import { getAgentServerWorkingDir } from "./agent-server-config";
import { getEffectiveLocalBackend } from "./backend-registry/active-store";
import { getAgentServerClientOptions } from "./agent-server-client-options";

/**
 * Hooks are optional, but the lookup sits on the conversation-start critical
 * path, so it has to fail fast rather than hold the user's launch open.
 * `loadHooks()` spends this budget twice — a `/server_info` version probe, then
 * `POST /api/hooks` — and the SDK's 60s default would stall a launch for two
 * minutes against a reachable-but-wedged agent-server.
 */
const HOOKS_LOAD_TIMEOUT_MS = 5000;

class HooksService {
  /**
   * Load workspace hooks from the agent-server by calling POST /api/hooks.
   * Returns the HookConfig if the workspace has hooks, or null.
   * Gracefully returns null when no usable local backend is available
   * (cloud backend, unseeded registry, or older agent-server).
   *
   * `projectDir` must be the workspace root, not a conversation's own working
   * dir: the agent-server checks `<project_dir>/.openhands/hooks.json` literally
   * and never walks parents.
   */
  static async loadWorkspaceHooks(
    projectDir?: string,
  ): Promise<HookConfig | null> {
    // getEffectiveLocalBackend() returns null for cloud backends AND for the
    // NO_BACKEND sentinel (unseeded/empty registry), avoiding the silent
    // NoBackendAvailableError that getActiveBackend() would cause.
    if (!getEffectiveLocalBackend()) {
      return null;
    }

    try {
      const response = await new HooksClient(
        getAgentServerClientOptions({ timeout: HOOKS_LOAD_TIMEOUT_MS }),
      ).loadHooks({
        project_dir: projectDir ?? getAgentServerWorkingDir(),
      });
      return response?.hook_config ?? null;
    } catch (error) {
      // Agent-server may not support the hooks endpoint or may be
      // unreachable; gracefully fall back to null. Log it, or a workspace
      // whose hooks silently never run leaves no diagnostic at all.
      console.warn(
        "Failed to load workspace hooks, continuing without:",
        error,
      );
      return null;
    }
  }
}

export default HooksService;
