#!/usr/bin/env node
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../../..",
);

const WORKFLOWS = [
  {
    id: "mock-llm-e2e",
    name: "Mock-LLM E2E Tests",
    path: ".github/workflows/mock-llm-e2e.yml",
    job: "mock-llm-e2e",
  },
  {
    id: "mock-llm-docker-e2e",
    name: "Mock-LLM Docker E2E Tests",
    path: ".github/workflows/mock-llm-docker-e2e.yml",
    job: "mock-llm-docker-e2e",
  },
];

const REPOSITORY = "OpenHands/agent-canvas";
const ORDINARY_BRANCH = "agent/e2e-only-on-main";
const RELEASE_BRANCH =
  "release-please--branches--main--components--agent-canvas";

function getByPath(root, dottedPath) {
  return dottedPath.split(".").reduce((value, key) => value?.[key], root);
}

function startsWith(value, prefix) {
  return String(value ?? "").startsWith(prefix);
}

function tokenize(expression) {
  const tokens = [];
  const matcher =
    /\s+|('(?:[^'\\]|\\.)*')|([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)|(&&|\|\||==|!=|[(),])/gy;
  let index = 0;

  while (index < expression.length) {
    matcher.lastIndex = index;
    const match = matcher.exec(expression);
    if (!match) {
      throw new Error(
        `Unsupported expression token near: ${expression.slice(index)}`,
      );
    }

    index = matcher.lastIndex;
    if (match[1]) tokens.push({ type: "string", value: match[1] });
    else if (match[2]) tokens.push({ type: "identifier", value: match[2] });
    else if (match[3]) tokens.push({ type: "operator", value: match[3] });
  }

  return tokens;
}

function evaluateExpression(expression, context) {
  const javascriptExpression = tokenize(expression)
    .map((token) => {
      if (token.type !== "identifier") return token.value;
      if (["startsWith", "true", "false", "null"].includes(token.value)) {
        return token.value;
      }
      return `getByPath(context, ${JSON.stringify(token.value)})`;
    })
    .join(" ");

  return Boolean(
    Function(
      "context",
      "getByPath",
      "startsWith",
      `return (${javascriptExpression});`,
    )(context, getByPath, startsWith),
  );
}

function extractBlock(text, startPattern, stopPattern = /^\S/m) {
  const startMatch = startPattern.exec(text);
  if (!startMatch) return "";
  const start = startMatch.index + startMatch[0].length;
  const remainder = text.slice(start);
  const stopMatch = stopPattern.exec(remainder);
  return stopMatch ? remainder.slice(0, stopMatch.index) : remainder;
}

function extractOnBlock(workflowText) {
  return extractBlock(workflowText, /^on:\n/m, /^\S/m);
}

function extractJobBlock(workflowText, jobName) {
  const jobsBlock = extractBlock(workflowText, /^jobs:\n/m, /^\S/m);
  return extractBlock(
    jobsBlock,
    new RegExp(`^  ${jobName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}:\\n`, "m"),
    /^  [A-Za-z0-9_-]+:\n/m,
  );
}

function extractJobIfExpression(workflowText, jobName) {
  const jobBlock = extractJobBlock(workflowText, jobName);
  const lines = jobBlock.split("\n");
  const ifIndex = lines.findIndex((line) => line.startsWith("    if:"));
  if (ifIndex === -1) return "true";

  const ifLine = lines[ifIndex];
  if (!ifLine.includes(">-")) {
    return ifLine.replace(/^\s*if:\s*/, "").trim();
  }

  const expressionLines = [];
  for (const line of lines.slice(ifIndex + 1)) {
    if (/^\s{6,}\S/.test(line)) {
      expressionLines.push(line.trim());
      continue;
    }
    break;
  }

  if (expressionLines.length === 0) {
    throw new Error(`Could not read folded if expression for ${jobName}`);
  }

  return expressionLines.join(" ");
}

function extractEventBlock(onBlock, eventName) {
  return extractBlock(
    onBlock,
    new RegExp(`^  ${eventName}:.*\\n`, "m"),
    /^  [A-Za-z_]+:/m,
  );
}

function parseInlineList(value) {
  const match = /\[([^\]]*)\]/.exec(value);
  if (!match) return [];
  return match[1]
    .split(",")
    .map((entry) => entry.trim().replace(/^['"]|['"]$/g, ""))
    .filter(Boolean);
}

function parseNestedInlineList(block, key) {
  const match = new RegExp(`^    ${key}:\\s*(\\[[^\\n]+\\])`, "m").exec(block);
  return match ? parseInlineList(match[1]) : [];
}

function branchFromRef(ref) {
  return ref?.startsWith("refs/heads/") ? ref.slice("refs/heads/".length) : ref;
}

function actionAllowed(block, action) {
  const types = parseNestedInlineList(block, "types");
  return types.length === 0 || types.includes(action);
}

function branchAllowed(block, branch) {
  const branches = parseNestedInlineList(block, "branches");
  return branches.length === 0 || branches.includes(branch);
}

function workflowTriggers(workflowText, context, action) {
  const onBlock = extractOnBlock(workflowText);
  const eventName = context.github.event_name;
  const eventBlock = extractEventBlock(onBlock, eventName);
  if (!eventBlock) return false;

  if (eventName === "push") {
    return branchAllowed(eventBlock, branchFromRef(context.github.ref));
  }

  if (eventName === "pull_request") {
    return actionAllowed(eventBlock, action);
  }

  if (eventName === "workflow_run") {
    const workflows = parseNestedInlineList(eventBlock, "workflows");
    const workflowName = context.github.event.workflow_run.name;
    return (
      (workflows.length === 0 || workflows.includes(workflowName)) &&
      actionAllowed(eventBlock, action) &&
      branchAllowed(eventBlock, context.github.event.workflow_run.head_branch)
    );
  }

  return eventName === "workflow_dispatch";
}

function buildContext({ eventName, ref, headRef, headRepo, workflowRun }) {
  return {
    github: {
      event_name: eventName,
      repository: REPOSITORY,
      ref,
      head_ref: headRef,
      sha: "0000000000000000000000000000000000000000",
      event: {
        pull_request: {
          number: 1834,
          head: {
            ref: headRef,
            sha: "1111111111111111111111111111111111111111",
            repo: { full_name: headRepo },
          },
        },
        workflow_run: workflowRun,
      },
    },
  };
}

const SCENARIOS = [
  {
    workflow: "mock-llm-e2e",
    scenario: "main push",
    action: "push",
    context: buildContext({ eventName: "push", ref: "refs/heads/main" }),
    expected: { workflowTriggered: true, jobRuns: true },
  },
  {
    workflow: "mock-llm-e2e",
    scenario: "non-main push",
    action: "push",
    context: buildContext({
      eventName: "push",
      ref: `refs/heads/${ORDINARY_BRANCH}`,
    }),
    expected: { workflowTriggered: false, jobRuns: false },
  },
  {
    workflow: "mock-llm-e2e",
    scenario: "ordinary same-repository PR",
    action: "synchronize",
    context: buildContext({
      eventName: "pull_request",
      headRef: ORDINARY_BRANCH,
      headRepo: REPOSITORY,
    }),
    expected: { workflowTriggered: true, jobRuns: false },
  },
  {
    workflow: "mock-llm-e2e",
    scenario: "same-repository release-please PR",
    action: "synchronize",
    context: buildContext({
      eventName: "pull_request",
      headRef: RELEASE_BRANCH,
      headRepo: REPOSITORY,
    }),
    expected: { workflowTriggered: true, jobRuns: true },
  },
  {
    workflow: "mock-llm-e2e",
    scenario: "fork release-pattern PR",
    action: "synchronize",
    context: buildContext({
      eventName: "pull_request",
      headRef: RELEASE_BRANCH,
      headRepo: "someone/agent-canvas",
    }),
    expected: { workflowTriggered: true, jobRuns: false },
  },
  {
    workflow: "mock-llm-e2e",
    scenario: "manual dispatch",
    action: "workflow_dispatch",
    context: buildContext({
      eventName: "workflow_dispatch",
      ref: "refs/heads/main",
    }),
    expected: { workflowTriggered: true, jobRuns: true },
  },
  {
    workflow: "mock-llm-docker-e2e",
    scenario: "successful Docker workflow_run on main",
    action: "completed",
    context: buildContext({
      eventName: "workflow_run",
      workflowRun: {
        name: "Docker",
        conclusion: "success",
        head_branch: "main",
        head_sha: "2222222222222222222222222222222222222222",
      },
    }),
    expected: { workflowTriggered: true, jobRuns: true },
  },
  {
    workflow: "mock-llm-docker-e2e",
    scenario: "failed Docker workflow_run on main",
    action: "completed",
    context: buildContext({
      eventName: "workflow_run",
      workflowRun: {
        name: "Docker",
        conclusion: "failure",
        head_branch: "main",
        head_sha: "3333333333333333333333333333333333333333",
      },
    }),
    expected: { workflowTriggered: true, jobRuns: false },
  },
  {
    workflow: "mock-llm-docker-e2e",
    scenario: "successful Docker workflow_run on non-main",
    action: "completed",
    context: buildContext({
      eventName: "workflow_run",
      workflowRun: {
        name: "Docker",
        conclusion: "success",
        head_branch: ORDINARY_BRANCH,
        head_sha: "4444444444444444444444444444444444444444",
      },
    }),
    expected: { workflowTriggered: false, jobRuns: false },
  },
  {
    workflow: "mock-llm-docker-e2e",
    scenario: "ordinary same-repository PR",
    action: "synchronize",
    context: buildContext({
      eventName: "pull_request",
      headRef: ORDINARY_BRANCH,
      headRepo: REPOSITORY,
    }),
    expected: { workflowTriggered: true, jobRuns: false },
  },
  {
    workflow: "mock-llm-docker-e2e",
    scenario: "same-repository release-please PR",
    action: "synchronize",
    context: buildContext({
      eventName: "pull_request",
      headRef: RELEASE_BRANCH,
      headRepo: REPOSITORY,
    }),
    expected: { workflowTriggered: true, jobRuns: true },
  },
  {
    workflow: "mock-llm-docker-e2e",
    scenario: "manual dispatch",
    action: "workflow_dispatch",
    context: buildContext({
      eventName: "workflow_dispatch",
      ref: "refs/heads/main",
    }),
    expected: { workflowTriggered: true, jobRuns: true },
  },
];

function evaluateScenario(workflow, scenario) {
  const workflowText = readFileSync(
    path.join(repoRoot, workflow.path),
    "utf-8",
  );
  const jobExpression = extractJobIfExpression(workflowText, workflow.job);
  const workflowTriggered = workflowTriggers(
    workflowText,
    scenario.context,
    scenario.action,
  );
  const jobConditionMatches = evaluateExpression(
    jobExpression,
    scenario.context,
  );
  const jobRuns = workflowTriggered && jobConditionMatches;

  return {
    workflow: workflow.id,
    job: workflow.job,
    scenario: scenario.scenario,
    event: scenario.context.github.event_name,
    workflowTriggered,
    jobConditionMatches,
    jobRuns,
    expected: scenario.expected,
    matchesExpected:
      workflowTriggered === scenario.expected.workflowTriggered &&
      jobRuns === scenario.expected.jobRuns,
  };
}

export function evaluateWorkflowPolicy() {
  return SCENARIOS.map((scenario) => {
    const workflow = WORKFLOWS.find(({ id }) => id === scenario.workflow);
    if (!workflow) throw new Error(`Unknown workflow ${scenario.workflow}`);
    return evaluateScenario(workflow, scenario);
  });
}

function printMarkdown(rows) {
  console.log(
    "Workflow event/condition evaluation against actual workflow files",
  );
  console.log("");
  console.log(`Repository: ${repoRoot}`);
  console.log(`Date: ${new Date().toISOString()}`);
  console.log("");
  console.log(
    "Limitations: this evaluates checked-in workflow triggers and job-level `if` expressions with representative GitHub event payloads. It does not start a GitHub-hosted runner, pull Docker images, execute workflow steps, or prove GitHub service-side scheduling beyond the modeled event payloads.",
  );
  console.log("");
  console.log(
    "| Workflow | Scenario | Event | Workflow triggered | Job condition | Job runs | Matches expected |",
  );
  console.log("| --- | --- | --- | --- | --- | --- | --- |");
  for (const row of rows) {
    console.log(
      `| ${row.workflow} | ${row.scenario} | ${row.event} | ${row.workflowTriggered ? "yes" : "no"} | ${row.jobConditionMatches ? "true" : "false"} | ${row.jobRuns ? "yes" : "no"} | ${row.matchesExpected ? "yes" : "NO"} |`,
    );
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const rows = evaluateWorkflowPolicy();
  if (process.argv.includes("--json")) {
    console.log(JSON.stringify(rows, null, 2));
  } else {
    printMarkdown(rows);
  }

  if (rows.some((row) => !row.matchesExpected)) {
    process.exitCode = 1;
  }
}
