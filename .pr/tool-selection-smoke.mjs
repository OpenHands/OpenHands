import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  AgentProfilesClient,
  AgentServerClient,
  ConversationClient,
  ProfilesClient,
} from "@openhands/typescript-client/clients";

assert.equal(
  process.env.TOOL_SELECTION_ISOLATED,
  "1",
  "Use a disposable stack; see .pr/tool-selection-validation.md",
);
const host = process.env.TOOL_SELECTION_SERVER_URL ?? "http://127.0.0.1:18500";
const mockUrl = process.env.TOOL_SELECTION_MOCK_URL ?? "http://127.0.0.1:18504";
for (const url of [host, mockUrl]) {
  assert(
    ["localhost", "127.0.0.1"].includes(new URL(url).hostname),
    "Only local test servers are supported",
  );
}
const options = {
  host,
  apiKey: readFileSync(
    process.env.TOOL_SELECTION_KEY_FILE ??
      "/private/tmp/tool-selection-live/api-key.txt",
    "utf8",
  ).trim(),
};
const workspace =
  process.env.TOOL_SELECTION_WORKSPACE ??
  "/private/tmp/tool-selection-live/workspace";
const prefix = `review-${Date.now()}`;
const profiles = new AgentProfilesClient(options);
const server = new AgentServerClient(options);
const conversations = new ConversationClient(options);
await new ProfilesClient(options).saveProfile(prefix, {
  llm: {
    model: "openai/mock-test-model",
    base_url: `${mockUrl}/v1`,
    api_key: "mock-test-key",
  },
  include_secrets: true,
});
async function mockPost(path, body = {}) {
  const response = await fetch(`${mockUrl}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  assert(response.ok);
}

const uiProfile = process.env.TOOL_SELECTION_UI_PROFILE;
if (uiProfile) {
  assert.deepEqual(
    (await profiles.getAgentProfile(uiProfile)).profile.tools,
    [],
  );
  console.log("UI save/reopen: persisted tools=[]");
}

for (const [name, inputTools, expectedSpecs] of [
  ["smoke-empty", [], []],
  ["smoke-glob", [{ name: "glob", params: {} }], ["glob"]],
  ["smoke-standard", null, null],
]) {
  const profile = {
    name: `${prefix}-${name}`,
    agent_kind: "openhands",
    llm_profile_ref: prefix,
    tools: inputTools,
    mcp_server_refs: [],
    secret_refs: [],
    condenser: { enabled: false },
  };
  await profiles.saveAgentProfile(profile.name, profile);
  const saved = (await profiles.getAgentProfile(profile.name)).profile;
  const materialized = await server.post(
    `/api/agent-profiles/${profile.name}/materialize`,
    { profile: saved },
  );
  const specs = materialized.resolved_settings.tools.map((tool) => tool.name);
  if (expectedSpecs) assert.deepEqual(specs, expectedSpecs);
  else {
    assert(specs.includes("terminal"));
    assert(specs.includes("switch_llm"));
  }

  await mockPost("/admin/reset");
  await mockPost("/admin/trajectory/register", {
    name: "finish-only",
    turns: Array.from({ length: 3 }, () => ({
      tool_call: {
        name: "finish",
        arguments: { message: "TOOL_SELECTION_OK" },
      },
    })),
  });
  await mockPost("/admin/trajectory/activate", { name: "finish-only" });
  const created = await conversations.createConversation({
    agent_profile_id: saved.id,
    title: name,
    workspace: {
      working_dir: `${workspace}/${profile.name}`,
    },
    max_iterations: 2,
    initial_message: {
      role: "user",
      content: [{ type: "text", text: "Finish with TOOL_SELECTION_OK." }],
    },
  });
  let state;
  for (let attempt = 0; attempt < 60; attempt++) {
    state = await conversations.getConversation(created.id);
    if (["finished", "error", "stuck"].includes(state.execution_status)) break;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  assert.equal(state.execution_status, "finished");
  const requests = await (await fetch(`${mockUrl}/admin/requests`)).json();
  const calls = Array.isArray(requests) ? requests : requests.requests;
  assert(calls.length > 0);
  const actual = calls.at(-1).tools.map((tool) => tool.function.name);
  assert.equal(new Set(actual).size, actual.length);
  assert(actual.includes("finish"));
  if (expectedSpecs?.length === 0) {
    assert(
      !actual.includes("terminal") &&
        !actual.includes("glob") &&
        !actual.includes("switch_llm"),
    );
  } else if (name === "smoke-glob") {
    assert(actual.includes("glob"));
    assert(!actual.includes("terminal") && !actual.includes("switch_llm"));
  } else {
    assert(actual.includes("switch_llm"));
  }
  console.log(
    JSON.stringify({
      name,
      conversation: created.id,
      specs,
      modelTools: actual,
      status: state.execution_status,
    }),
  );
}
