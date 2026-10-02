/**
 * Reference-chain validator for Deep Planning (#17819).
 *
 * Documents in the chain cite upstream sections inline as `[Req 3.1]`,
 * `[DB 2.1]`, `[BE 4.2]`, `[FE 1.1]`. A chain is valid when every citation
 * resolves to a section that actually exists in an *upstream* document, and
 * every requirement is picked up by at least one task.
 *
 * Kept pure (no React, no store) so it can be unit-tested in both directions —
 * a validator that only ever returns `ok` would be indistinguishable from one
 * that is not running at all.
 */

import {
  DEEP_PLAN_LABEL_TO_PHASE,
  DEEP_PLAN_PHASE_IDS,
  DEEP_PLAN_PHASES,
  type DeepPlanLabel,
  type DeepPlanPhaseId,
  getDeepPlanPhase,
  upstreamPhasesOf,
} from "#/utils/deep-plan";

export type RefIssueReason =
  /** The cited section does not exist in the upstream document. */
  | "dangling"
  /** The cited label belongs to a phase that is not upstream of this document. */
  | "not-upstream"
  /** The cited label has no document in the chain at all. */
  | "missing-document";

export interface RefIssue {
  /** Document the bad citation appears in. */
  from: DeepPlanPhaseId;
  /** The citation as written, e.g. `DB 2.5.1`. */
  ref: string;
  reason: RefIssueReason;
}

export interface RefReport {
  ok: boolean;
  issues: RefIssue[];
  /** Requirements sections no task cites. */
  uncovered: string[];
}

const REFERENCE_PATTERN = /\[(Req|DB|BE|FE)\s+([0-9]+(?:\.[0-9]+)*)\]/g;
const HEADING_PATTERN = /^#{1,6}\s+(.*)$/;
const NUMBER_TOKEN = /\b[0-9]+(?:\.[0-9]+)*\b/g;
/** A bracketed citation, e.g. `[Req 3.1]`, removed before scanning a heading. */
const CITATION_IN_TEXT = /\[(?:Req|DB|BE|FE)\s+[0-9]+(?:\.[0-9]+)*\]/g;
/**
 * A heading that spells its own label out at the start, e.g.
 * `## [Req 3.1] Users` — the citation *is* the section number here.
 */
const OWN_LABEL_HEADING = /^\s*\[(?:Req|DB|BE|FE)\s+([0-9]+(?:\.[0-9]+)*)\]/;

export type DeepPlanDocuments = Partial<Record<DeepPlanPhaseId, string>>;

/**
 * Section numbers a document defines, taken from its numbered headings
 * (`## 3.1 Authentication` defines `3.1`; `### 3.1.1 Login` defines `3.1.1`).
 * A heading may also spell its own label out (`## [Req 3.1] Authentication`).
 *
 * Upstream citations are not definitions. In `## 2.1 Users [Req 3.1]` the
 * document defines `2.1`, and `3.1` is the requirement it satisfies — counting
 * it as defined here would let a downstream `[DB 3.1]` resolve against a
 * section this document never defines. The citation's number is only a
 * definition when the heading *leads* with it (`## [Req 3.1] Users`); an
 * unnumbered heading that merely cites upstream (`## Users [Req 3.1]`) defines
 * nothing.
 */
export function extractDefinedSections(content: string): Set<string> {
  const sections = new Set<string>();
  for (const line of content.split("\n")) {
    const heading = HEADING_PATTERN.exec(line);
    if (!heading) continue;
    const title = heading[1];
    const withoutCitations = title.replace(CITATION_IN_TEXT, " ");
    const tokens = withoutCitations.match(NUMBER_TOKEN);
    if (tokens) {
      for (const token of tokens) {
        sections.add(token);
      }
      continue;
    }
    const ownLabel = OWN_LABEL_HEADING.exec(title);
    if (ownLabel) sections.add(ownLabel[1]);
  }
  return sections;
}

/** Every citation in a document, in order of appearance. */
export function extractReferences(
  content: string,
): { label: DeepPlanLabel; section: string }[] {
  const refs: { label: DeepPlanLabel; section: string }[] = [];
  for (const match of content.matchAll(REFERENCE_PATTERN)) {
    refs.push({ label: match[1] as DeepPlanLabel, section: match[2] });
  }
  return refs;
}

export function validateDocumentChain(
  documents: DeepPlanDocuments,
  /**
   * Only validate documents up to and including this phase. A checkpoint
   * validates the document it is confirming; later documents are validated at
   * their own checkpoints. Validating the whole chain here would let a stale
   * citation in a document the user has not reached block an upstream
   * reconfirmation they have no way to repair from that checkpoint.
   */
  throughPhase?: DeepPlanPhaseId,
): RefReport {
  const issues: RefIssue[] = [];
  const sectionsByPhase = new Map<DeepPlanPhaseId, Set<string>>();

  const lastIndex = throughPhase
    ? DEEP_PLAN_PHASE_IDS.indexOf(throughPhase)
    : DEEP_PLAN_PHASE_IDS.length - 1;

  for (const phase of DEEP_PLAN_PHASES) {
    const content = documents[phase.id];
    if (content !== undefined) {
      sectionsByPhase.set(phase.id, extractDefinedSections(content));
    }
  }

  for (const phase of DEEP_PLAN_PHASES) {
    const content = documents[phase.id];
    if (content === undefined) continue;
    if (DEEP_PLAN_PHASE_IDS.indexOf(phase.id) > lastIndex) continue;

    const upstream = new Set(upstreamPhasesOf(phase.id));

    for (const { label, section } of extractReferences(content)) {
      const ref = `${label} ${section}`;
      const target = DEEP_PLAN_LABEL_TO_PHASE[label];

      if (!upstream.has(target)) {
        issues.push({ from: phase.id, ref, reason: "not-upstream" });
        continue;
      }

      const defined = sectionsByPhase.get(target);
      if (!defined) {
        issues.push({ from: phase.id, ref, reason: "missing-document" });
        continue;
      }

      if (!defined.has(section)) {
        issues.push({ from: phase.id, ref, reason: "dangling" });
      }
    }
  }

  // Requirements coverage: every requirement section must be cited by a task.
  // Only meaningful once the tasks document is inside the validated range;
  // before that, "uncovered" would flag requirements the user simply has not
  // written tasks for yet.
  const uncovered: string[] = [];
  const requirementSections = sectionsByPhase.get("requirements");
  const tasks = documents.tasks;
  const tasksInRange = DEEP_PLAN_PHASE_IDS.indexOf("tasks") <= lastIndex;
  if (requirementSections && tasks !== undefined && tasksInRange) {
    const citedByTasks = new Set(
      extractReferences(tasks)
        .filter((ref) => ref.label === "Req")
        .map((ref) => ref.section),
    );
    for (const section of requirementSections) {
      if (!citedByTasks.has(section)) uncovered.push(section);
    }
    uncovered.sort();
  }

  return { ok: issues.length === 0, issues, uncovered };
}

/** Human-readable reason, for the checkpoint error message. */
export const describeRefIssue = (issue: RefIssue): string => {
  const phase = getDeepPlanPhase(issue.from);
  switch (issue.reason) {
    case "dangling":
      return `${phase.outputFile ?? issue.from} cites [${issue.ref}], which no upstream document defines.`;
    case "not-upstream":
      return `${phase.outputFile ?? issue.from} cites [${issue.ref}], which is not an upstream document for this phase.`;
    case "missing-document":
      return `${phase.outputFile ?? issue.from} cites [${issue.ref}], but that document has not been produced yet.`;
  }
};
