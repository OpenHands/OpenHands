// Known bugs as executable expected failures.
//
// A feature-map note that records an open bug may carry a fenced block
//
//   ```sh known-bug F05.example desktop
//   control-openhands browser goto /
//   control-openhands assert --field enabled --equals true --bug-shows false -- browser enabled 'testid=submit-button'  # bug
//   ```
//
// whose commands assert the correct behavior. The one command ending in
// `# bug` (it may continue over lines ending in `\`) is a
// `control-openhands assert` that names what the bug shows: `--bug-shows V`
// for the asserted field's value, or `--bug 'PATH OP [VALUE] && ...'` for
// clauses on any fields of the command's output. `bug run` executes the
// block and classifies each attempt:
// the bug reproduced (xfail) only when that assertion is the command that
// failed and it saw the bug's own value; the correct behavior held (xpass);
// anything else (setup, timeout, an unhealthy stack, another symptom) is a
// fail. `bug proof` turns two runs' attempts into a verdict on a fix.
import { createHash } from "node:crypto";

const OPEN = /^(\s*)```sh known-bug (F\d{2}\.[a-z0-9-]+)(?: ([a-z0-9-]+))?\s*$/;
const BUG_LINE = /#\s*bug\s*$/;
const BUG_OPS = [
  "eq",
  "ne",
  "contains",
  "not-contains",
  "matches",
  "exists",
  "missing",
];
const CLAUSE = new RegExp(`^(\\S+)\\s+(${BUG_OPS.join("|")})(?:\\s+(.*))?$`);
const unquote = (v) =>
  /^(['"]).*\1$/s.test(v) && v.length >= 2 ? v.slice(1, -1) : v;

/**
 * A `--bug` predicate: `PATH OP [VALUE]` clauses joined by `&&`, the grammar
 * the sibling verification skills share. OP is one of eq, ne, contains,
 * not-contains, matches (a regex), exists, missing; a VALUE with spaces may
 * be quoted.
 * @returns {{ clauses: {path: string, op: string, value?: string}[], problems: string[] }}
 */
export function parseBugPredicate(text) {
  const clauses = [];
  const problems = [];
  for (const raw of String(text).split("&&")) {
    const part = raw.trim();
    const m = CLAUSE.exec(part);
    if (!m) {
      problems.push(
        `\`${part}\` is not PATH OP [VALUE] with OP in ${BUG_OPS.join(", ")}`,
      );
      continue;
    }
    const [, path, op, value] = m;
    const takesValue = op !== "exists" && op !== "missing";
    if (takesValue && value === undefined) {
      problems.push(`\`${part}\` needs a value (quote an empty one: '')`);
      continue;
    }
    if (!takesValue && value !== undefined) {
      problems.push(`\`${part}\`: ${op} takes no value`);
      continue;
    }
    const clause = takesValue
      ? { path, op, value: unquote(value.trim()) }
      : { path, op };
    if (op === "matches")
      try {
        new RegExp(clause.value);
      } catch (e) {
        problems.push(`\`${part}\`: ${e.message}`);
        continue;
      }
    clauses.push(clause);
  }
  return { clauses, problems };
}

/** Whether one clause holds on a command's JSON output. */
export function clauseHolds(json, { path, op, value }) {
  const actual = fieldValue(json, path);
  if (op === "exists") return actual !== undefined;
  if (op === "missing") return actual === undefined;
  if (actual === undefined) return false;
  if (op === "eq" || op === "ne") {
    const same =
      actual === value ||
      (value.trim() !== "" &&
        actual.trim() !== "" &&
        Number.isFinite(Number(actual)) &&
        Number(actual) === Number(value));
    return op === "eq" ? same : !same;
  }
  if (op === "contains") return actual.includes(value);
  if (op === "not-contains") return !actual.includes(value);
  return new RegExp(value).test(actual);
}

// The value of a `--bug` flag on a recipe line, for static checks.
function bugFlag(command) {
  const m = /--bug(?:=|\s+)('[^']*'|"[^"]*"|\S+)/.exec(command);
  return m ? unquote(m[1]) : undefined;
}

/**
 * Every known-bug block in a map file's text.
 * @returns {{ id: string, name: string, line: number, script: string, bugLines: number[], bugRanges: [number, number][], sha: string, problems: string[] }[]}
 */
export function knownBugBlocks(text) {
  const lines = text.split("\n");
  const blocks = [];
  for (let i = 0; i < lines.length; i += 1) {
    const open = OPEN.exec(lines[i]);
    if (!open) continue;
    const [, indent, id, name = "default"] = open;
    const body = [];
    let j = i + 1;
    while (j < lines.length && lines[j].trim() !== "```") {
      body.push(
        lines[j].startsWith(indent) ? lines[j].slice(indent.length) : lines[j],
      );
      j += 1;
    }
    const problems = [];
    if (j >= lines.length) problems.push("the block is never closed");
    const bugLines = body
      .map((l, n) => (BUG_LINE.test(l) ? n + 1 : 0))
      .filter(Boolean);
    if (bugLines.length !== 1)
      problems.push(
        `needs exactly one line ending in \`# bug\` (found ${bugLines.length})`,
      );
    // A command continued with `\` starts at the first line of the
    // continuation; bash reports that line when it fails.
    const bugRanges = bugLines.map((n) => {
      let start = n;
      while (start > 1 && /\\$/.test(body[start - 2])) start -= 1;
      return [start, n];
    });
    for (const [start, n] of bugRanges) {
      const command = body
        .slice(start - 1, n)
        .map((l) => l.replace(/\\$/, ""))
        .join(" ");
      if (!/^control-openhands assert\b/.test(command.trim())) {
        problems.push(
          `line ${n}: the \`# bug\` line must be a \`control-openhands assert\``,
        );
        continue;
      }
      const predicate = bugFlag(command);
      if (predicate === undefined && !/--bug-shows\b/.test(command))
        problems.push(
          `line ${n}: the \`# bug\` assertion must name what the bug shows with --bug-shows or --bug`,
        );
      else if (predicate !== undefined)
        for (const p of parseBugPredicate(predicate).problems)
          problems.push(`line ${n}: --bug ${p}`);
    }
    const script = body.join("\n");
    blocks.push({
      id,
      name,
      line: i + 1,
      script,
      bugLines,
      bugRanges,
      sha: createHash("sha256").update(script).digest("hex").slice(0, 16),
      problems,
    });
    i = j;
  }
  return blocks;
}

/**
 * The value at a dotted path of a command's JSON output, as a string:
 * strings as they are, everything else as JSON.
 */
export function fieldValue(json, path) {
  let value = json;
  for (const key of String(path).split(".")) {
    if (value === null || typeof value !== "object" || !(key in value))
      return undefined;
    value = value[key];
  }
  return typeof value === "string" ? value : JSON.stringify(value);
}

/**
 * What an assertion saw.
 * kind: hold | mismatch (the bug's own symptom: every --bug-shows/--bug
 * clause holds, or any mismatch when none is given) | other-symptom (a
 * mismatch, but not the bug's) | error (the command failed or printed no such
 * field).
 */
export function judgeAssertion({
  exitCode,
  json,
  field,
  equals,
  matches,
  bugShows,
  bug,
}) {
  if (exitCode !== 0 || !json)
    return { kind: "error", reason: `the command exited ${exitCode}` };
  const actual = fieldValue(json, field);
  if (actual === undefined)
    return { kind: "error", reason: `the output has no field ${field}` };
  const holds =
    equals !== undefined
      ? actual === String(equals)
      : new RegExp(matches).test(actual);
  if (holds) return { kind: "hold", actual };
  const clauses = [
    ...(bugShows !== undefined
      ? [{ path: field, op: "eq", value: String(bugShows) }]
      : []),
    ...(bug !== undefined ? parseBugPredicate(bug).clauses : []),
  ];
  if (!clauses.length) return { kind: "mismatch", actual };
  const bugChecks = clauses.map((c) => ({
    clause: `${c.path} ${c.op}${c.value === undefined ? "" : ` ${JSON.stringify(c.value)}`}`,
    actual: fieldValue(json, c.path),
    ok: clauseHolds(json, c),
  }));
  return {
    kind: bugChecks.every((c) => c.ok) ? "mismatch" : "other-symptom",
    actual,
    bugChecks,
  };
}

/**
 * One attempt's result.
 * @param {object} a
 * @param {boolean} a.doctorBefore  false when the stack was unhealthy before the attempt
 * @param {boolean|undefined} a.doctorAfter  false when it was unhealthy after a failure
 * @param {number} a.exitCode
 * @param {boolean} a.timedOut
 * @param {number|null} a.failedLine  1-based line of the block that failed
 * @param {number[]} a.bugLines
 * @param {[number, number][]} [a.bugRanges]  the marked command's first and last line
 * @param {{kind: string, line?: number}|undefined} a.assertion  the assertion logged at the marked command
 * @returns {{ status: "xfail"|"xpass"|"fail", reason?: string }}
 */
export function classifyAttempt(a) {
  if (a.doctorBefore === false)
    return {
      status: "fail",
      reason: "setup: doctor failed before the attempt",
    };
  if (a.timedOut) return { status: "fail", reason: "timeout" };
  if (a.exitCode === 0) return { status: "xpass" };
  if (a.failedLine == null)
    return { status: "fail", reason: "the failing line is unknown" };
  const [start, end] = a.bugRanges?.[0] ?? [a.bugLines[0], a.bugLines[0]];
  if (a.failedLine < start)
    return {
      status: "fail",
      reason: `setup: line ${a.failedLine} failed before the \`# bug\` line`,
    };
  if (a.failedLine > end)
    return {
      status: "fail",
      reason: `line ${a.failedLine} failed after the \`# bug\` assertion held`,
    };
  if (a.doctorAfter === false)
    return {
      status: "fail",
      reason: "backend: doctor failed after the attempt",
    };
  const kind = a.assertion?.kind;
  if (kind === "mismatch") return { status: "xfail" };
  if (kind === "other-symptom")
    return {
      status: "fail",
      reason:
        "wrong symptom: the assertion saw neither the expected nor the bug's value",
    };
  if (kind === "error")
    return {
      status: "fail",
      reason: "the `# bug` command itself failed; no value was compared",
    };
  return {
    status: "fail",
    reason: "no assertion result was recorded at the `# bug` line",
  };
}

const same = (values) => new Set(values).size <= 1;

/**
 * Verdict on a fix from the attempts recorded on a baseline run and a fix
 * run for one block. Only attempts of the given recipe (sha) count.
 * verified: every baseline attempt (at least two) reproduced the bug and
 * every fix attempt (at least two) passed, under matching inputs.
 * not-fixed: the fix still reproduces it. inconclusive: anything else.
 * `vary` names what the fix changed: "revision" (default, a Canvas fix: same
 * backend, different revisions) or "backend" (an Agent Server or automation
 * fix: same revision, a different backend version on each side).
 */
export function proofVerdict({
  baseline,
  fix,
  sha,
  min = 2,
  vary = "revision",
}) {
  const reasons = [];
  const pick = (rows) => rows.filter((r) => r.sha === sha);
  const b = pick(baseline);
  const f = pick(fix);
  const ignored = baseline.length + fix.length - b.length - f.length;
  const tally = (rows) =>
    rows.reduce((t, r) => ({ ...t, [r.status]: (t[r.status] ?? 0) + 1 }), {});
  const bx = b.filter((r) => r.status === "xfail").length;
  const fp = f.filter((r) => r.status === "xpass").length;
  if (b.length < min)
    reasons.push(
      `baseline has ${b.length} attempt(s) of this recipe; need ${min}`,
    );
  else if (bx !== b.length)
    reasons.push(
      `baseline reproduced the bug in ${bx} of ${b.length} attempts (flaky or not reproduced)`,
    );
  let notFixed = false;
  if (f.length < min)
    reasons.push(`fix has ${f.length} attempt(s) of this recipe; need ${min}`);
  else if (fp !== f.length) {
    notFixed = f.some((r) => r.status === "xfail");
    reasons.push(`fix passed in ${fp} of ${f.length} attempts`);
  }
  const all = [...b, ...f];
  const inputs = [];
  if (!same(all.map((r) => r.viewport ?? "")))
    inputs.push("attempts ran at different viewports");
  if (vary === "backend") {
    if (!same(all.map((r) => r.revision)))
      inputs.push("attempts ran different revisions");
    if (
      !same(b.map((r) => r.backend ?? "")) ||
      !same(f.map((r) => r.backend ?? ""))
    )
      inputs.push("attempts on one side ran different backend versions");
    if (b.length && f.length && same(all.map((r) => r.backend ?? "")))
      inputs.push("baseline and fix ran the same backend versions");
  } else {
    if (!same(all.map((r) => r.backend ?? "")))
      inputs.push("attempts ran against different backend versions");
    if (b.length && f.length && same(all.map((r) => r.revision)))
      inputs.push("baseline and fix ran the same revision");
    if (!same(b.map((r) => r.revision)) || !same(f.map((r) => r.revision)))
      inputs.push("attempts on one side ran different revisions");
  }
  reasons.push(...inputs);
  // Mismatched inputs or an unreliable baseline make any fix reading moot.
  const baselineReproduced = b.length >= min && bx === b.length;
  const verdict = !reasons.length
    ? "verified"
    : notFixed && baselineReproduced && !inputs.length
      ? "not-fixed"
      : "inconclusive";
  return {
    verdict,
    reasons,
    baseline: tally(b),
    fix: tally(f),
    ignored,
  };
}

/**
 * The viewport to put back after an attempt, as `browser viewport` takes it
 * (`WxH`), or null when the block left it as it found it. A block may switch
 * to phone (F07.overview-toggle phone-overview); without the restore the next
 * attempt, and the caller's next check, would silently run at phone width.
 */
export function viewportToRestore(before, after) {
  if (!before?.width || !before?.height) return null;
  if (after?.width === before.width && after?.height === before.height)
    return null;
  return `${before.width}x${before.height}`;
}
