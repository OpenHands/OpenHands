#!/usr/bin/env node
// Temporary patch for @openhands/extensions until the change is released upstream.
//
// gitlab-issue-to-mr: when the automation names an agent profile, start the
// local coding conversation from that profile (for example Claude Code through
// ACP) instead of a plain agent built from the global LLM setting. Without a
// profile the script behaves as before. Same change as the teammate fix on
// sumitvairagar/extensions, branch factory-local, commit cfb567c8e.
//
// The script is stored twice in the package: as the skill's main.py and inside
// automations/bundle-index.js (as an escaped string). Both are patched. If the
// expected code is not found (for example after a package upgrade), the script
// fails so the build stops instead of silently shipping unpatched code.
//
// Usage: node scripts/patch-extensions.mjs [path to node_modules/@openhands/extensions]

import fs from "node:fs";
import path from "node:path";

const root = path.resolve(
  process.argv[2] ?? "node_modules/@openhands/extensions",
);

const OLD = `    payload: dict = {
        "workspace": {"working_dir": str(workspace_dir)},
        "agent": _get_agent_dict(agent_url, api_key),
        "initial_message": {"content": [{"text": initial_message}]},
    }
`;

const NEW = `    payload: dict = {
        "workspace": {"working_dir": str(workspace_dir)},
        "initial_message": {"content": [{"text": initial_message}]},
    }
    # The automation service sets the variable only when the automation names
    # an agent profile; the Agent Server launches that profile, as the cloud
    # path does. Without one, the conversation uses the settings' LLM.
    profile_id = os.environ.get("AUTOMATION_AGENT_PROFILE_ID")
    if profile_id:
        payload["agent_profile_id"] = profile_id
    else:
        payload["agent"] = _get_agent_dict(agent_url, api_key)
`;

const MARKER = 'payload["agent_profile_id"] = profile_id';

function fail(message) {
  console.error(`patch-extensions: ${message}`);
  process.exit(1);
}

function count(text, part) {
  return text.split(part).length - 1;
}

const scriptPath = path.join(
  root,
  "skills/gitlab-issue-to-mr/scripts/main.py",
);
const bundlePath = path.join(root, "automations/bundle-index.js");

const original = fs.readFileSync(scriptPath, "utf8");
if (original.includes(MARKER)) {
  console.log("patch-extensions: gitlab-issue-to-mr already patched, nothing to do");
  process.exit(0);
}
if (count(original, OLD) !== 1) {
  fail(`expected code not found exactly once in ${scriptPath}`);
}
const patched = original.replace(OLD, NEW);

// The bundle stores the whole file as a JSON-escaped string.
const bundle = fs.readFileSync(bundlePath, "utf8");
const escapedOriginal = JSON.stringify(original).slice(1, -1);
if (count(bundle, escapedOriginal) !== 1) {
  fail(`gitlab-issue-to-mr script not found exactly once in ${bundlePath}`);
}
const escapedPatched = JSON.stringify(patched).slice(1, -1);

fs.writeFileSync(scriptPath, patched);
fs.writeFileSync(bundlePath, bundle.replace(escapedOriginal, () => escapedPatched));
console.log("patch-extensions: patched gitlab-issue-to-mr (main.py and bundle-index.js)");
