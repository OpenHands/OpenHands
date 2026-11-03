// Run from the performance checkout: node .pr/chat-performance/summary-benchmark.mjs
// CPU microbenchmark only; it does not measure browser input or layout.
import { createRequire } from "node:module";
import { execFileSync } from "node:child_process";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const root = process.cwd();
const require = createRequire(path.join(root, "package.json"));
const { build } = require("esbuild");
const base = "dce9d017abe9ed3d362bc77e0cd3fff2e4931f22";
const source = "src/components/conversation-events/chat/event-message-components/event-group-summary.ts";
const output = await mkdtemp(path.join(tmpdir(), "openhands-lookup-bench-"));
await build({
  stdin: {
    contents: execFileSync("git", ["show", `${base}:${source}`], { encoding: "utf8" }),
    resolveDir: path.join(root, path.dirname(source)),
    loader: "ts",
  },
  tsconfig: path.join(root, "tsconfig.json"),
  bundle: true,
  platform: "node",
  format: "esm",
  outfile: path.join(output, "before.mjs"),
});
await build({ entryPoints: [source], bundle: true, platform: "node", format: "esm", outfile: path.join(output, "after.mjs") });
const before = await import(pathToFileURL(path.join(output, "before.mjs")));
const after = await import(pathToFileURL(path.join(output, "after.mjs")));
const results = [];
for (const count of [500, 1000, 2000, 4000]) {
  const all = [];
  const observations = [];
  for (let index = 0; index < count; index += 1) {
    const action = {
      id: `a${index}`, timestamp: "2026-11-03T00:00:00Z", source: "agent", thought: [], thinking_blocks: [],
      action: { kind: "ExecuteBashAction", command: "true", is_input: false, timeout: null, reset: false },
      tool_name: "execute_bash", tool_call_id: `t${index}`,
      tool_call: { id: `t${index}`, type: "function", function: { name: "execute_bash", arguments: "{}" } },
      llm_response_id: `l${index}`, security_risk: "UNKNOWN",
    };
    const observation = {
      id: `o${index}`, timestamp: "2026-11-03T00:00:00Z", source: "environment", action_id: action.id,
      tool_name: action.tool_name, tool_call_id: action.tool_call_id,
      observation: { kind: "ExecuteBashObservation", content: [{ type: "text", text: "ok" }], command: "true", exit_code: 0, error: false, timeout: false, metadata: {} },
    };
    all.push(action, observation);
    observations.push(observation);
  }
  const result = { actionObservationPairs: count };
  for (const [label, fn] of [["beforeMs", before.getEventGroupSummaryParts], ["afterMs", after.getEventGroupSummaryParts]]) {
    fn(observations, all);
    const times = [];
    for (let repetition = 0; repetition < 7; repetition += 1) {
      const start = performance.now();
      const parts = fn(observations, all);
      if (parts[0]?.count !== count) throw new Error("Summary count changed");
      times.push(performance.now() - start);
    }
    times.sort((a, b) => a - b);
    result[label] = Math.round(times[3] * 100) / 100;
  }
  results.push(result);
}
const report = { base, method: "median of seven calls after one warmup; includes after-index construction", results };
await writeFile(path.join(output, "results.json"), JSON.stringify(report, null, 2));
console.log(JSON.stringify({ ...report, artifacts: output }, null, 2));
