import { useMutation } from "@tanstack/react-query";
import { useRef } from "react";

import type { CommandResult } from "#/api/runtime-service/agent-server-runtime-service";
import { DEFAULT_WORKING_DIR } from "#/api/agent-server-config";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";
import { useBashCommandRunner } from "#/hooks/use-bash-command-runner";
import type {
  SecurityScanError,
  SecurityScanResult,
} from "#/types/security-scan";
import {
  OPENGREP_SCAN_COMMAND,
  parseOpengrepJsonOutput,
} from "#/utils/opengrep-output";

const SCAN_TIMEOUT_SEC = 300;

type RunCommand = (
  command: string,
  cwd: string,
  timeout: number,
) => Promise<CommandResult>;

function mapScanFailure(result: CommandResult): SecurityScanError {
  if (result.exit_code === 127) {
    return {
      code: "opengrep_not_installed",
      message: result.stderr.trim() || "opengrep_not_installed",
    };
  }

  return {
    code: "scan_failed",
    message: result.stderr.trim() || `exit_code_${result.exit_code}`,
  };
}

async function runSecuritySastScan(
  run: RunCommand,
  workingDir: string,
): Promise<SecurityScanResult> {
  const result = await run(OPENGREP_SCAN_COMMAND, workingDir, SCAN_TIMEOUT_SEC);

  if (result.exit_code === 127) {
    throw mapScanFailure(result);
  }

  const stdout = result.stdout.trim();
  if (!stdout) {
    if (result.exit_code !== 0) {
      throw mapScanFailure(result);
    }
    return {
      findings: [],
      scannedAt: new Date().toISOString(),
      tool: "opengrep",
    };
  }

  try {
    return parseOpengrepJsonOutput(stdout);
  } catch {
    throw {
      code: "invalid_output",
      message: result.stderr.trim() || "invalid_output",
    } satisfies SecurityScanError;
  }
}

export function useSecuritySastScan() {
  const { data: conversation } = useActiveConversation();
  const runtimeIsReady = useRuntimeIsReady();

  const conversationUrl = conversation?.conversation_url;
  const sessionApiKey = conversation?.session_api_key;
  const workingDir =
    conversation?.workspace?.working_dir?.trim() || DEFAULT_WORKING_DIR;

  const bashEnabled =
    runtimeIsReady && !!conversationUrl && !!sessionApiKey && !!workingDir;

  const runCommand = useBashCommandRunner(
    conversationUrl,
    sessionApiKey,
    bashEnabled,
  );

  const runCommandRef = useRef(runCommand);
  runCommandRef.current = runCommand;

  return useMutation<SecurityScanResult, SecurityScanError>({
    mutationKey: ["security-sast-scan", conversation?.id, workingDir],
    mutationFn: async () => {
      if (!bashEnabled) {
        throw {
          code: "runtime_unavailable",
          message: "runtime_unavailable",
        } satisfies SecurityScanError;
      }

      const run: RunCommand = (command, cwd, timeout) =>
        runCommandRef.current(command, cwd, timeout);

      return runSecuritySastScan(run, workingDir);
    },
    meta: { disableToast: true },
  });
}
