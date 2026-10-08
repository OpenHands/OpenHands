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
/**
 * A heading that leads with its section number, e.g. `## 3.1 Authentication`
 * defines `3.1`. Only a *leading* number counts: a title like
 * `## 3.1 Response under 200 ms` defines `3.1`, not `200`, and
 * `## Version 1.2` defines nothing.
 */
const LEADING_NUMBER_HEADING = /^\s*([0-9]+(?:\.[0-9]+)*)\b/;
/**
 * A heading that spells its own label out at the start, e.g.
 * `## [Req 3.1] Users` — the citation *is* the section number here.
 */
const OWN_LABEL_HEADING = /^\s*\[(?:Req|DB|BE|FE)\s+([0-9]+(?:\.[0-9]+)*)\]/;

export type DeepPlanDocuments = Partial<Record<DeepPlanPhaseId, string>>;

/**
 * Section numbers a document defines, taken from its numbered headings
 * (`## 3.1 Authentication` defines `3.1`; `### 3.1.1 Login` defines `3.1.1`).
 * Only a number that *leads* the heading is a definition, so prose numbers in
 * a title (`## 3.1 Response under 200 ms` → `3.1`, not `200`) and version-like
 * titles (`## Version 1.2` → nothing) are ignored.
 *
 * A heading may instead spell its own label out (`## [Req 3.1] Authentication`),
 * in which case the leading citation is the definition. An unnumbered heading
 * that merely cites upstream (`## Users [Req 3.1]`) defines nothing — counting
 * it would let a downstream `[DB 3.1]` resolve against a section this document
 * never defines.
 */
export function extractDefinedSections(content: string): Set<string> {
  const sections = new Set<string>();
  for (const line of content.split("\n")) {
    const heading = HEADING_PATTERN.exec(line);
    if (!heading) continue;
    const title = heading[1];
    const leadingNumber = LEADING_NUMBER_HEADING.exec(title);
    if (leadingNumber) {
      sections.add(leadingNumber[1]);
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
    uncovered.sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  }

  return { ok: issues.length === 0, issues, uncovered };
}
