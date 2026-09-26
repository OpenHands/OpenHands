/**
 * Structured root cause analysis (RCA) context handed to a coding task by an
 * external system (e.g. HolmesGPT or any other investigation tool). Canvas
 * normalizes the input into {@link RcaContext} and renders it into the first
 * user message via {@link buildRcaContextPrompt} so the agent analyzes the
 * repository against the RCA before making changes.
 */
export interface RcaContext {
  /** Short description of the diagnosed root cause. Required. */
  summary: string;
  /** Supporting observations: log lines, metrics, alerts, stack traces. */
  evidence: string[];
  /** Services, modules, or components suspected of containing the fault. */
  suspectedComponents: string[];
  /** Repository paths suspected of containing the fault. */
  suspectedFiles: string[];
  /** Remediation the RCA suggests, if any. */
  recommendedAction?: string;
  /** Free-form origin label (e.g. "holmesgpt"); attribution only. */
  source?: string;
}

function toTrimmedString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function toStringList(value: unknown): string[] {
  const items = Array.isArray(value) ? value : [value];
  return items
    .map((item) => toTrimmedString(item))
    .filter((item): item is string => item !== undefined);
}

function pick(record: Record<string, unknown>, ...keys: string[]): unknown {
  for (const key of keys) {
    if (key in record) return record[key];
  }
  return undefined;
}

/**
 * Normalize an untrusted external payload into {@link RcaContext}, accepting
 * both snake_case and camelCase field names so producers are not locked to a
 * naming convention. Returns `null` when the payload is unusable (not an
 * object or missing a non-empty `summary`).
 */
export function parseRcaContext(input: unknown): RcaContext | null {
  if (typeof input !== "object" || input === null || Array.isArray(input)) {
    return null;
  }
  const record = input as Record<string, unknown>;

  const summary = toTrimmedString(record.summary);
  if (!summary) return null;

  return {
    summary,
    evidence: toStringList(record.evidence),
    suspectedComponents: toStringList(
      pick(record, "suspected_components", "suspectedComponents"),
    ),
    suspectedFiles: toStringList(
      pick(record, "suspected_files", "suspectedFiles"),
    ),
    recommendedAction: toTrimmedString(
      pick(record, "recommended_action", "recommendedAction"),
    ),
    source: toTrimmedString(record.source),
  };
}

/**
 * Decode the base64-encoded JSON `rca` deep-link parameter into an
 * {@link RcaContext}. Inverse of the encoding produced by
 * `buildRcaLaunchPath` in `#/utils/rca-launch-url`.
 */
export function decodeRcaParam(encoded: string): RcaContext | null {
  try {
    return parseRcaContext(JSON.parse(atob(encoded)));
  } catch {
    return null;
  }
}

/**
 * Render an {@link RcaContext} as a markdown block for the agent's first user
 * message. Written for the agent, not the UI, so it stays English-only and is
 * deliberately phrased to make the agent validate the RCA against the actual
 * repository rather than trusting it blindly.
 */
export function buildRcaContextPrompt(rca: RcaContext): string {
  const lines = [
    "## Root Cause Analysis Context",
    "",
    `The following root cause analysis was provided${
      rca.source ? ` by ${rca.source}` : " by an external system"
    } for this task. Treat it as a hypothesis to verify against the ` +
      "repository, not as established fact.",
    "",
    "### Summary",
    rca.summary,
  ];

  const listSection = (title: string, items: string[]) => {
    if (items.length === 0) return;
    lines.push("", `### ${title}`, ...items.map((item) => `- ${item}`));
  };

  listSection("Evidence", rca.evidence);
  listSection("Suspected Components", rca.suspectedComponents);
  listSection("Suspected Files", rca.suspectedFiles);

  if (rca.recommendedAction) {
    lines.push("", "### Recommended Action", rca.recommendedAction);
  }

  lines.push(
    "",
    "Before making changes, analyze the repository using this context: " +
      "inspect the suspected files and components, verify the evidence " +
      "against the actual code, and confirm or refine the root cause. " +
      "Then implement the recommended action, or a better-supported fix " +
      "if the analysis does not hold.",
  );

  return lines.join("\n");
}

/**
 * Compose the task text for the agent's first user message: the RCA context
 * block first (so it reads as supplied context), then the user's own task
 * text. Returns `undefined` when neither input carries content.
 */
export function composeTaskWithRcaContext(
  taskText: string | undefined,
  rca: RcaContext | null | undefined,
): string | undefined {
  const parts = [
    rca ? buildRcaContextPrompt(rca) : undefined,
    taskText?.trim() || undefined,
  ].filter((part): part is string => Boolean(part));
  return parts.length > 0 ? parts.join("\n\n") : undefined;
}
