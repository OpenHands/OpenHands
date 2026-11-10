import type { ACPAgentProfile } from "@openhands/typescript-client";
import {
  ACP_CUSTOM_PRESET_KEY,
  getAcpProvider,
  type ACPProviderConfig,
} from "#/constants/acp-providers";
import { parseCommand } from "#/utils/acp-command";

/**
 * The launch configuration an ACP profile (or the Settings draft) would use.
 *
 * `command` and `args` mirror the agent-server's `ACPAgentSettings` fields and
 * are sent separately, so a profile that stores `acp_args` keeps them apart.
 * {@link acpLaunchArgv} joins them for the one thing that must treat the launch
 * as a single identity: the model-discovery cache key. The agent-server runs
 * `command[0]` as the executable and passes `command.slice(1)` followed by
 * `args`, so the joined argv identifies the same process either way.
 */
export interface AcpLaunch {
  /** A built-in provider key, the `"custom"` sentinel, or a forward-compat key. */
  server: string;
  /** Explicit argv, already tokenized. Empty means "use the server's default". */
  command: readonly string[];
  /** Extra arguments appended after `command`. */
  args: readonly string[];
}

/** The full argv for `launch` — its identity for caches and remembered lists. */
export function acpLaunchArgv(launch: AcpLaunch): string[] {
  return [...launch.command, ...launch.args];
}

/**
 * The launch a saved ACP profile describes.
 *
 * Built-in providers resolve their own default command in the agent-server, so
 * their launch carries no explicit argv. A profile read from the profiles API
 * may still include `acp_command` (the default one, or an edited one); keeping
 * it would change the discovery payload and the remembered-list scope for a
 * provider that previously sent only its server key. Customized built-in
 * commands are out of scope for pre-conversation discovery, so they are
 * deliberately ignored here too.
 *
 * A custom profile keeps its command: `acp_command` holds one shell string
 * parsed with {@link parseCommand}, while profiles created through the API may
 * instead carry the extra tokens in `acp_args`.
 */
export function acpLaunchFromProfile(
  profile:
    | Pick<ACPAgentProfile, "acp_server" | "acp_command" | "acp_args">
    | null
    | undefined,
): AcpLaunch | null {
  if (!profile?.acp_server) return null;
  if (getAcpProvider(profile.acp_server)) {
    return { server: profile.acp_server, command: [], args: [] };
  }
  return {
    server: profile.acp_server,
    command: profile.acp_command ? parseCommand(profile.acp_command) : [],
    args: profile.acp_args ?? [],
  };
}

/**
 * The preset a command line matches, or {@link ACP_CUSTOM_PRESET_KEY}.
 *
 * The match is on the tokenized command, so quoting and extra whitespace do not
 * turn a built-in provider into a custom one.
 */
export function detectAcpPreset(
  commandText: string,
  providers: readonly ACPProviderConfig[],
): string {
  const normalized = parseCommand(commandText).join(" ");
  const match = providers.find(
    (provider) => normalized === provider.default_command.join(" "),
  );
  return match?.key ?? ACP_CUSTOM_PRESET_KEY;
}

/**
 * The launch the Settings command draft describes.
 *
 * A built-in provider always asks with its registry default (no explicit
 * command); a custom command goes out as the tokens the user typed. An empty
 * custom command has nothing to launch, so it returns null.
 */
export function acpLaunchFromCommandText(
  commandText: string,
  providers: readonly ACPProviderConfig[],
): AcpLaunch | null {
  const preset = detectAcpPreset(commandText, providers);
  if (preset !== ACP_CUSTOM_PRESET_KEY) {
    return { server: preset, command: [], args: [] };
  }
  const tokens = parseCommand(commandText);
  return tokens.length
    ? { server: ACP_CUSTOM_PRESET_KEY, command: tokens, args: [] }
    : null;
}

/**
 * Whether asking the agent-server for this launch's models makes sense.
 *
 * A built-in provider resolves its own default command, so an empty argv is
 * fine. The `"custom"` sentinel (and any key Canvas does not know) has no
 * default, so an empty argv would only produce a 422 — skip the request.
 */
export function isAcpLaunchDiscoverable(
  launch: AcpLaunch | null | undefined,
): launch is AcpLaunch {
  if (!launch) return false;
  return acpLaunchArgv(launch).length > 0 || !!getAcpProvider(launch.server);
}
