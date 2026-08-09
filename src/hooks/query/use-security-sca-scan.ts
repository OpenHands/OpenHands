import { useMutation } from "@tanstack/react-query";
import { useRef } from "react";

import type { CommandResult } from "#/api/runtime-service/agent-server-runtime-service";
import { DEFAULT_WORKING_DIR } from "#/api/agent-server-config";
import { DependencyTrackService } from "#/api/integrations/dependency-track-service";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useConversationDependencyTrackIntegration } from "#/hooks/query/use-dependency-track-integration";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";
import { useBashCommandRunner } from "#/hooks/use-bash-command-runner";
import type { ScaScanError, ScaScanResult } from "#/types/security-scan";
import { mapDependencyTrackFindings } from "#/utils/dependency-track-findings";
import { SYFT_SCAN_COMMAND } from "#/utils/syft-output";

const SCAN_TIMEOUT_SEC = 300;

type RunCommand = (
  command: string,
  cwd: string,
  timeout: number,
) => Promise<CommandResult>;

function mapSyftFailure(result: CommandResult): ScaScanError {
  if (result.exit_code === 127) {
    return {
      code: "syft_not_installed",
      message: result.stderr.trim() || "syft_not_installed",
    };
  }
  return {
    code: "scan_failed",
    message: result.stderr.trim() || `exit_code_${result.exit_code}`,
  };
}

async function generateSbom(
  run: RunCommand,
  workingDir: string,
): Promise<string> {
  const result = await run(SYFT_SCAN_COMMAND, workingDir, SCAN_TIMEOUT_SEC);
  if (result.exit_code === 127) {
    throw mapSyftFailure(result);
  }
  const stdout = result.stdout.trim();
  if (!stdout) {
    throw {
      code: "invalid_output",
      message: result.stderr.trim() || "empty_sbom",
    } satisfies ScaScanError;
  }
  try {
    JSON.parse(stdout);
  } catch {
    throw {
      code: "invalid_output",
      message: "invalid_sbom_json",
    } satisfies ScaScanError;
  }
  return stdout;
}

export function useSecurityScaScan() {
  const { data: conversation } = useActiveConversation();
  const runtimeIsReady = useRuntimeIsReady();
  const dtIntegration = useConversationDependencyTrackIntegration();

  const conversationUrl = conversation?.conversation_url;
  const sessionApiKey = conversation?.session_api_key;
  const workingDir =
    conversation?.workspace?.working_dir?.trim() || DEFAULT_WORKING_DIR;
  const workspaceId = dtIntegration.workspaceId;
  const projectUuid = dtIntegration.config.projectUuid;

  const bashEnabled =
    runtimeIsReady &&
    !!conversationUrl &&
    !!sessionApiKey &&
    !!workingDir &&
    dtIntegration.isReady &&
    !!workspaceId;

  const runCommand = useBashCommandRunner(
    conversationUrl,
    sessionApiKey,
    bashEnabled,
  );

  const runCommandRef = useRef(runCommand);
  runCommandRef.current = runCommand;

  return useMutation<ScaScanResult, ScaScanError>({
    mutationKey: [
      "security-sca-scan",
      conversation?.id,
      workingDir,
      workspaceId,
      projectUuid,
    ],
    mutationFn: async () => {
      if (!runtimeIsReady || !conversationUrl || !sessionApiKey || !workingDir) {
        throw {
          code: "runtime_unavailable",
          message: "runtime_unavailable",
        } satisfies ScaScanError;
      }
      if (!dtIntegration.isReady || !workspaceId || !projectUuid) {
        throw {
          code: "dependency_track_not_configured",
          message: "dependency_track_not_configured",
        } satisfies ScaScanError;
      }

      const run: RunCommand = (command, cwd, timeout) =>
        runCommandRef.current(command, cwd, timeout);

      const bomJson = await generateSbom(run, workingDir);
      const client = DependencyTrackService.forWorkspace(workspaceId);

      let token: string;
      try {
        token = await client.uploadBom(bomJson, projectUuid);
      } catch (error) {
        throw {
          code: "bom_upload_failed",
          message: error instanceof Error ? error.message : "bom_upload_failed",
        } satisfies ScaScanError;
      }

      try {
        await client.waitForBomProcessing(token);
      } catch (error) {
        throw {
          code: "bom_processing_failed",
          message:
            error instanceof Error ? error.message : "bom_processing_failed",
        } satisfies ScaScanError;
      }

      const findings = await client.listProjectFindings(projectUuid);

      return {
        findings: mapDependencyTrackFindings(findings),
        scannedAt: new Date().toISOString(),
        tool: "dependency-track",
      };
    },
    meta: { disableToast: true },
  });
}
