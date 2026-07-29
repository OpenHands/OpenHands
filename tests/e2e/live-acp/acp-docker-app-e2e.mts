/**
 * Live e2e through the APPLICATION's own conversation-start path (not the
 * isolated request builder). Drives the same functions the running app calls:
 *
 *   onboarding "Set up credentials"  -> SecretsService.createSecret(name, value)
 *   choose-agent step                -> buildAcpAgentSettingsDiff(...) PATCH /api/settings
 *   conversation start               -> buildStartConversationRequestWithEncryptedSettings(...)
 *                                         (reads settings + the saved secret NAMES,
 *                                          emits each as a LookupSecret the
 *                                          agent-server resolves from its store)
 *
 * This is the piece the request-builder script can't cover: it proves the saved
 * credentials round-trip through the backend store and the orchestrator emits
 * the right LookupSecrets, end-to-end, against a real container.
 *
 * Requires an agent-server with software-agent-sdk#3510 (first in v1.25.0) —
 * ACP credentials resolve off the event loop; an older image deadlocks.
 *
 *   npx vite-node -c tests/e2e/live-acp/vite-node.config.mts \
 *     tests/e2e/live-acp/acp-docker-app-e2e.mts -- codex
 *
 * One provider per process (settings are global on the backend; a fresh process
 * avoids the SettingsService cache bleeding between providers).
 *
 * The provider plans, host credential collectors, and HTTP/poll helpers are
 * shared with the request-builder script — see ./harness.mts.
 */
import { SecretsService } from "#/api/secrets-service";
import { buildStartConversationRequestWithEncryptedSettings } from "#/api/agent-server-adapter";
import { buildAcpAgentSettingsDiff } from "#/constants/acp-providers";
import { SettingsClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import {
  BASE,
  fetchFinalReply,
  getProviderPlan,
  pollUntilTerminal,
  postJson,
  registerDockerBackend,
  type ProviderPlan,
} from "./harness.mts";

registerDockerBackend();

async function run(plan: ProviderPlan): Promise<boolean> {
  const secrets = plan.collectSecrets();
  if (!secrets) {
    console.log(`⏭️  ${plan.id}: SKIP — credentials not present on host`);
    return true;
  }

  const settingsClient = new SettingsClient(getAgentServerClientOptions());

  // 1) onboarding "Set up credentials" — save each container cred as a global
  //    secret via the SAME service call the SetupAcpSecretsStep makes.
  console.log(
    `▶️  ${plan.id}: saving container creds via SecretsService.createSecret [${Object.keys(
      secrets,
    ).join(", ")}]`,
  );
  for (const [name, value] of Object.entries(secrets)) {
    await SecretsService.createSecret(name, value);
  }

  // 2) choose-agent step — persist ACP agent settings via the app's diff builder.
  const diff = buildAcpAgentSettingsDiff(plan.acpServer, { model: plan.model });
  if (!diff) throw new Error(`no settings diff for ${plan.acpServer}`);
  if (plan.sessionMode) diff.acp_session_mode = plan.sessionMode;
  await settingsClient.updateSettings({ agent_settings_diff: diff });
  console.log(
    `   PATCHed agent settings: ${JSON.stringify({
      acp_server: diff.acp_server,
      acp_model: diff.acp_model,
      ...(plan.sessionMode ? { acp_session_mode: plan.sessionMode } : {}),
    })}`,
  );

  // 3) conversation start — the app's own orchestrator. It re-reads settings +
  //    the saved secret names and emits each as a LookupSecret.
  const workingDir = `/workspace/app-e2e/${plan.id}-${Date.now()}`;
  const payload = (await buildStartConversationRequestWithEncryptedSettings({
    settings: undefined as any, // base settings come from the backend fetch
    query: `Reply with exactly: ${plan.expectedToken}`,
    workingDir,
  })) as any;

  const emitted = payload.secrets ?? {};
  console.log(
    `   orchestrator emitted secrets: ${Object.entries(emitted)
      .map(([k, v]: any) => `${k}=${v.kind}`)
      .join(", ")}`,
  );
  // Assert the orchestrator emitted a LookupSecret for each saved credential.
  const missing = Object.keys(secrets).filter(
    (n) => emitted[n]?.kind !== "LookupSecret",
  );
  if (missing.length > 0) {
    console.log(
      `   ❌ ${plan.id}: orchestrator did NOT emit a LookupSecret for: ${missing.join(", ")}`,
    );
    return false;
  }

  const created = await postJson(`${BASE}/api/conversations`, payload);
  const id = created.id;
  console.log(`   conversation ${id} created; polling…`);

  const status = await pollUntilTerminal(id);
  const reply = await fetchFinalReply(id);
  const ok = reply.includes(plan.expectedToken);
  console.log(
    `   status=${status} reply=${JSON.stringify(reply.slice(0, 160))}`,
  );
  console.log(`   ${ok ? "✅ PASS" : "❌ FAIL"} (expected "${plan.expectedToken}")`);
  return ok;
}

async function main() {
  const args = process.argv.slice(2).filter((a) => a !== "--");
  const plan = args[0] ? getProviderPlan(args[0]) : undefined;
  if (!plan) {
    console.error(`usage: ... acp-docker-app-e2e.mts -- <codex|claude|gemini>`);
    process.exit(2);
  }
  console.log(`App-path e2e against ${BASE} — provider: ${plan.id}`);
  const ok = await run(plan).catch((e) => {
    console.error(`   ❌ ${plan.id} errored: ${(e as Error).message}`);
    return false;
  });
  console.log(ok ? "\nPASS" : "\nFAIL");
  process.exit(ok ? 0 : 1);
}

main();
