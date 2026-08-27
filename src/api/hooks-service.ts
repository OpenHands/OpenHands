import { HooksClient } from "@openhands/typescript-client/clients";
import type { HookConfig } from "@openhands/typescript-client";
import { getAgentServerWorkingDir } from "./agent-server-config";
import { getActiveBackend } from "./backend-registry/active-store";
import { getAgentServerClientOptions } from "./agent-server-client-options";

class HooksService {
  /**
   * Load workspace hooks from the agent-server by calling POST /api/hooks.
   * Returns the HookConfig if the workspace has hooks, or null.
   * Gracefully returns null on errors (e.g. cloud backend or older agent-server).
   */
  static async loadWorkspaceHooks(
    projectDir?: string,
  ): Promise<HookConfig | null> {
    if (getActiveBackend().backend.kind === "cloud") {
      return null;
    }

    try {
      const response = await new HooksClient(
        getAgentServerClientOptions(),
      ).loadHooks({
        project_dir: projectDir ?? getAgentServerWorkingDir(),
      });
      return response?.hook_config ?? null;
    } catch {
      // Agent-server may not support the hooks endpoint or may be
      // unreachable; gracefully fall back to null.
      return null;
    }
  }
}

export default HooksService;
