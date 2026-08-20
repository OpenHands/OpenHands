/**
 * Live e2e for the containerized ACP path (agent-canvas#1013/#1014).
 *
 * Exercises CANVAS'S OWN code path — it saves each credential to the
 * agent-server's secret store via {@link SecretsService.createSecret} (exactly
 * as onboarding does), then imports {@link buildStartConversationRequest} and
 * builds each provider's start request as the app does, POSTs it to a real
 * agent-server container, and asserts a real agent reply. This is the
 * "it actually works" check the unit tests can't give: it proves the
 * LookupSecrets Canvas emits resolve back from the store and authenticate the
 * CLI end-to-end (including the SDK's acp_file_secrets materialisation).
 *
 * Requires agent-server v1.28.0: it includes both software-agent-sdk#3510 for
 * off-loop LookupSecret resolution and the client_tools API used by
 * canvas_ui_control. Older images either deadlock resolving ACP credentials or
 * omit the Canvas UI tool.
 *
 * Excluded from `npm test` (lives under tests/). Run it by hand against a
 * running container:
 *
 *   docker run -d --name oh-acp -p 8010:8000 -v oh-acp-data:/workspace \
 *     -v "$(pwd)/tools:/canvas-tools:ro" -e OH_EXTRA_PYTHON_PATH=/canvas-tools \
 *     ghcr.io/openhands/agent-server:1.28.0-python
 *   npx vite-node -c tests/e2e/live-acp/vite-node.config.mts \
 *     tests/e2e/live-acp/acp-docker-e2e.mts -- codex claude gemini
 *
 * The provider plans, host credential collectors, and HTTP/poll helpers are
 * shared with the app-path script — see ./harness.mts. A provider whose
 * credentials aren't present on the host is skipped.
 */
import { SecretsService } from "#/api/secrets-service";
import { buildStartConversationRequest } from "#/api/agent-server-adapter";
import { DEFAULT_SETTINGS } from "#/services/settings";
import {
  BASE,
  PROVIDER_PLANS,
  fetchFinalReply,
  pollUntilTerminal,
  postJson,
  registerDockerBackend,
  type ProviderId,
  type ProviderPlan,
} from "./harness.mts";

registerDockerBackend();

// Canvas gives every conversation its OWN working_dir (<base>/<id_hex>) so the
// agent-server can init a fresh git repo + worktree per conversation. Mirror
// that here with a unique dir per run/provider — sharing one dir makes the
// second `git worktree add` collide on the same repo.
const WORKING_DIR_BASE =
  process.env.ACP_E2E_WORKING_DIR_BASE ?? "/workspace/acp-e2e";

function buildRequest(
  plan: ProviderPlan,
  secretNames: string[],
  workingDir: string,
) {
  // Build via the same function the app uses — this is the whole point of the
  // exercise. Each saved credential is referenced by name as a LookupSecret; the
  // agent-server resolves the value back from its own store (where the
  // SecretsService.createSecret calls above put it) at spawn time.
  return buildStartConversationRequest({
    settings: {
      ...DEFAULT_SETTINGS,
      agent_settings: {
        ...DEFAULT_SETTINGS.agent_settings,
        agent_kind: "acp",
        acp_server: plan.acpServer,
        acp_model: plan.model,
        ...(plan.sessionMode ? { acp_session_mode: plan.sessionMode } : {}),
      },
      conversation_settings: {
        ...DEFAULT_SETTINGS.conversation_settings,
        max_iterations: 8,
      },
    },
    query: `Reply with exactly: ${plan.expectedToken}`,
    workingDir,
    customSecrets: secretNames.map((name) => ({ name })),
  });
}

async function runProvider(plan: ProviderPlan): Promise<boolean> {
  const secrets = plan.collectSecrets();
  if (!secrets) {
    console.log(`\n⏭️  ${plan.id}: SKIP — credentials not present on host`);
    return true; // skip is not a failure
  }
  console.log(
    `\n▶️  ${plan.id}: building request via buildStartConversationRequest ` +
      `(acp_server=${plan.acpServer}, acp_model=${plan.model}, ` +
      `secrets=[${Object.keys(secrets).join(", ")}])`,
  );

  // Onboarding step: save each credential to the agent-server's secret store so
  // the LookupSecret the start request emits can be resolved back.
  for (const [name, value] of Object.entries(secrets)) {
    await SecretsService.createSecret(name, value);
  }

  const workingDir = `${WORKING_DIR_BASE}/${plan.id}-${Date.now()}`;
  const payload = buildRequest(plan, Object.keys(secrets), workingDir);
  // Sanity-check the request the app would send, without leaking values.
  const emitted = payload.secrets as Record<string, { kind: string }>;
  console.log(
    `   emitted secret kinds: ${Object.entries(emitted ?? {})
      .map(([k, v]) => `${k}=${v.kind}`)
      .join(", ")}`,
  );
  const notLookup = Object.entries(emitted ?? {}).filter(
    ([, v]) => v.kind !== "LookupSecret",
  );
  if (notLookup.length > 0) {
    console.log(
      `   ❌ ${plan.id}: expected all secrets as LookupSecret, got ${notLookup
        .map(([k, v]) => `${k}=${v.kind}`)
        .join(", ")}`,
    );
    return false;
  }

  const created = await postJson(`${BASE}/api/conversations`, payload);
  const id = created.id;
  console.log(`   conversation ${id} created; polling…`);

  const status = await pollUntilTerminal(id);
  console.log(`   execution_status=${status}`);

  const reply = await fetchFinalReply(id);
  const ok = reply.includes(plan.expectedToken);
  console.log(
    `   reply: ${JSON.stringify(reply.slice(0, 200))}\n   ${
      ok ? "✅ PASS" : "❌ FAIL"
    } (expected to contain "${plan.expectedToken}")`,
  );
  if (!ok && plan.id === "gemini" && !plan.sessionMode && status === "error") {
    // The credential path (materialise ADC → vertex-ai auth) is what this PR
    // proves; gemini-cli ≥0.43 rejects the registry default session mode
    // ("yolo") during headless init — an SDK/gemini-cli issue, not a credential
    // one. Re-run with the override to confirm the full turn.
    console.log(
      "   ℹ️  Likely the SDK/gemini-cli set_session_mode('yolo') blocker, not a " +
        "credential problem. Re-run with ACP_E2E_GEMINI_SESSION_MODE=default to " +
        "confirm the credential path end-to-end.",
    );
  }
  return ok;
}

async function main() {
  const args = process.argv.slice(2).filter((a) => a !== "--");
  const selected =
    args.length > 0
      ? PROVIDER_PLANS.filter((p) => args.includes(p.id))
      : PROVIDER_PLANS;

  console.log(`ACP Docker e2e against ${BASE} — providers: ${selected
    .map((p) => p.id)
    .join(", ")}`);

  const results: Array<{ id: ProviderId; ok: boolean }> = [];
  for (const plan of selected) {
    try {
      results.push({ id: plan.id, ok: await runProvider(plan) });
    } catch (error) {
      console.log(`   ❌ ${plan.id} errored: ${(error as Error).message}`);
      results.push({ id: plan.id, ok: false });
    }
  }

  console.log("\n=== summary ===");
  for (const r of results) console.log(`  ${r.ok ? "✅" : "❌"} ${r.id}`);
  const failed = results.filter((r) => !r.ok);
  if (failed.length > 0) {
    console.log(`\n${failed.length} provider(s) failed.`);
    process.exit(1);
  }
  console.log("\nAll selected providers passed.");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
