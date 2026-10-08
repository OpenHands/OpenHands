import type { TFunction } from "i18next";
import { describe, expect, it } from "vitest";
import { I18nKey } from "#/i18n/declaration";
import {
  extractDefinedSections,
  extractReferences,
  validateDocumentChain,
  type DeepPlanDocuments,
} from "#/utils/deep-plan-reference";
import { makeRefIssueMessage } from "#/utils/deep-plan-messages";

const requirements = [
  "# Requirements",
  "",
  "## 1 Scope",
  "",
  "## 3.1 Authentication",
  "",
  "### 3.1.1 Login",
  "",
  "## 5.1 Reporting",
].join("\n");

const database = [
  "# Database design",
  "",
  "## 2.1 Users [Req 3.1]",
  "",
  "### 2.5.1 Sessions [Req 3.1.1]",
].join("\n");

const validChain = (): DeepPlanDocuments => ({
  requirements,
  database,
  backend: [
    "# Backend design",
    "",
    "## 4.2 Login endpoint [Req 3.1] [DB 2.5.1]",
  ].join("\n"),
  frontend: [
    "# Frontend design",
    "",
    "## 1.1 Login form [Req 3.1] [DB 2.1] [BE 4.2]",
  ].join("\n"),
  tasks: [
    "# Tasks",
    "",
    "- [ ] T1 Login [Req 3.1] [DB 2.1] [BE 4.2] [FE 1.1]",
    "- [ ] T2 Sessions [Req 3.1.1]",
    "- [ ] T3 Reporting [Req 5.1]",
    "- [ ] T4 Scope [Req 1]",
  ].join("\n"),
});

describe("extractDefinedSections", () => {
  it("collects the numbered headings a document defines", () => {
    expect([...extractDefinedSections(requirements)].sort()).toEqual([
      "1",
      "3.1",
      "3.1.1",
      "5.1",
    ]);
  });

  it("ignores body text that is not a heading", () => {
    expect([...extractDefinedSections("See 9.9 for details.")]).toEqual([]);
  });

  it("does not count a heading's upstream citation as a defined section", () => {
    // `## 2.1 Users [Req 3.1]` defines 2.1 only. Counting 3.1 as well would let
    // a downstream `[DB 3.1]` resolve against a section the database document
    // never defines, defeating the whole traceability guarantee.
    expect([...extractDefinedSections("## 2.1 Users [Req 3.1]")]).toEqual([
      "2.1",
    ]);
  });

  it("still reads a section from a heading that spells its own label out", () => {
    expect([...extractDefinedSections("## [Req 3.1] Users")]).toEqual(["3.1"]);
  });

  it("only counts a heading's leading number, not prose numbers in the title", () => {
    // `200` is an SLA figure, not a section the document defines; counting it
    // would let `[Req 200]` resolve against a section that never exists.
    expect([...extractDefinedSections("## 3.1 Response under 200 ms")]).toEqual(
      ["3.1"],
    );
  });

  it("defines nothing for a heading that does not lead with a number", () => {
    expect([...extractDefinedSections("## Version 1.2")]).toEqual([]);
  });

  it("sees a leading number wrapped in supported inline Markdown", () => {
    // Markdown renders these wrappers without changing the displayed section
    // number, so a citation of the visible `3.1` must still resolve.
    expect([...extractDefinedSections("## **3.1** Authentication")]).toEqual([
      "3.1",
    ]);
    expect([...extractDefinedSections("## `3.1` Authentication")]).toEqual([
      "3.1",
    ]);
    expect([...extractDefinedSections("### *3.1.1* Login")]).toEqual(["3.1.1"]);
  });

  it("sees a leading number when a wrapper spans the whole title", () => {
    // The closing delimiter follows the title, not the number; the rendered
    // heading still starts with `3.1`.
    expect([...extractDefinedSections("## **3.1 Authentication**")]).toEqual([
      "3.1",
    ]);
    expect([...extractDefinedSections("## *3.1 Authentication*")]).toEqual([
      "3.1",
    ]);
    expect([...extractDefinedSections("## `3.1 Authentication`")]).toEqual([
      "3.1",
    ]);
  });

  it("does not treat a wrapped prose number as a definition", () => {
    // Wrapping does not lift the leading-token constraint: `**200**` is still
    // prose, not a section.
    expect([
      ...extractDefinedSections("## 3.1 Response under **200** ms"),
    ]).toEqual(["3.1"]);
  });

  it("ignores an unmatched Markdown marker before the number", () => {
    // An unclosed delimiter renders literally, so the displayed heading does
    // not start with the number and defines no section.
    expect([...extractDefinedSections("## *3.1 Authentication")]).toEqual([]);
    expect([...extractDefinedSections("## `3.1 Authentication")]).toEqual([]);
    expect([...extractDefinedSections("## **3.1 Authentication")]).toEqual([]);
    // `*3.1*` closes, so it is a real definition even with a trailing marker.
    expect([...extractDefinedSections("## *3.1* Authentication")]).toEqual([
      "3.1",
    ]);
  });

  it("does not define a section from an unnumbered heading's citation", () => {
    // `## Users [Req 3.1]` has no number of its own, so it defines nothing.
    // Treating the citation as a definition would let a downstream
    // `[DB 3.1]` resolve against a database section that never exists.
    expect([...extractDefinedSections("## Users [Req 3.1]")]).toEqual([]);
  });
});

describe("extractReferences", () => {
  it("reads every labelled citation in order", () => {
    expect(extractReferences("[Req 3.1] then [DB 2.5.1]")).toEqual([
      { label: "Req", section: "3.1" },
      { label: "DB", section: "2.5.1" },
    ]);
  });

  it("ignores labels outside the fixed vocabulary", () => {
    expect(extractReferences("[XX 1.1] [Req 3.1]")).toEqual([
      { label: "Req", section: "3.1" },
    ]);
  });
});

describe("validateDocumentChain", () => {
  it("accepts a chain whose every citation resolves upstream", () => {
    const report = validateDocumentChain(validChain());

    expect(report.issues).toEqual([]);
    expect(report.ok).toBe(true);
    expect(report.uncovered).toEqual([]);
  });

  it("rejects a dangling reference to a section that does not exist upstream", () => {
    const documents = validChain();
    documents.backend = "## 4.2 Login endpoint [Req 3.1] [DB 2.5.1] [DB 9.9.9]";

    const report = validateDocumentChain(documents);

    expect(report.ok).toBe(false);
    expect(report.issues).toEqual([
      { from: "backend", ref: "DB 9.9.9", reason: "dangling" },
    ]);
  });

  it("rejects a reference to a phase that is not upstream", () => {
    const documents = validChain();
    // The database design must not depend on the backend design. Drop the
    // downstream citations so this case isolates exactly one violation.
    documents.database = "## 2.1 Users [Req 3.1] [BE 4.2]";
    documents.backend = "# Backend design";
    documents.frontend = "# Frontend design";
    documents.tasks = "# Tasks";

    const report = validateDocumentChain(documents);

    expect(report.ok).toBe(false);
    expect(report.issues).toEqual([
      { from: "database", ref: "BE 4.2", reason: "not-upstream" },
    ]);
  });

  it("reports a citation to an upstream document that has not been produced", () => {
    const documents = validChain();
    // Drop the database document and the downstream chain with it, so only the
    // one citation to the missing document is in play.
    delete documents.database;
    documents.backend = "## 4.2 Login endpoint [Req 3.1] [DB 2.5.1]";
    documents.frontend = "# Frontend design";
    documents.tasks = "# Tasks";

    const report = validateDocumentChain(documents);

    expect(report.ok).toBe(false);
    expect(report.issues).toEqual([
      { from: "backend", ref: "DB 2.5.1", reason: "missing-document" },
    ]);
  });

  it("rejects a citation to a section only an unnumbered heading cites", () => {
    // Reviewer scenario: the database document's heading has no number of its
    // own — it only cites the requirement it satisfies. A backend citation of
    // `[DB 3.1]` must not resolve against that upstream citation.
    const documents: DeepPlanDocuments = {
      requirements: "## 3.1 Authentication",
      database: "## Users [Req 3.1]",
      backend: "## 4.1 API [DB 3.1]",
      tasks: "# Tasks",
    };

    const report = validateDocumentChain(documents);

    expect(report.ok).toBe(false);
    expect(report.issues).toEqual([
      { from: "backend", ref: "DB 3.1", reason: "dangling" },
    ]);
  });

  it("rejects a citation to a number that only appears inside a heading title", () => {
    // `200` is prose in the requirements heading (`under 200 ms`), not a
    // definition. Counting every number in a title would make `[Req 200]`
    // resolve, so the database must instead see it as dangling.
    const documents: DeepPlanDocuments = {
      requirements: "## 3.1 Latency under 200 ms",
      database: "## 2.1 X [Req 200]",
      tasks: "# Tasks",
    };

    const report = validateDocumentChain(documents);

    expect(report.issues).toEqual([
      { from: "database", ref: "Req 200", reason: "dangling" },
    ]);
  });

  it("resolves a citation to a requirement heading wrapped in Markdown", () => {
    // Reviewer scenario: `## **3.1** Authentication` displays section 3.1, so
    // the database's `[Req 3.1]` must resolve rather than block the checkpoint.
    const documents: DeepPlanDocuments = {
      requirements: "## **3.1** Authentication",
      database: "## 2.1 Users [Req 3.1]",
      tasks: "# Tasks",
    };

    const report = validateDocumentChain(documents);

    expect(report.issues).toEqual([]);
    expect(report.ok).toBe(true);
  });

  it("resolves a citation to a heading wrapped around the whole title", () => {
    // `## **3.1 Authentication**` (and code spans) render the number at the
    // start even though the closing delimiter follows the title.
    for (const requirements of [
      "## **3.1 Authentication**",
      "## `3.1 Authentication`",
    ]) {
      const report = validateDocumentChain({
        requirements,
        database: "## 2.1 Users [Req 3.1]",
        tasks: "# Tasks",
      });

      expect(report.issues).toEqual([]);
      expect(report.ok).toBe(true);
    }
  });

  it("reports a citation to a heading with an unmatched marker as dangling", () => {
    // `## *3.1 Authentication` shows the marker literally, so it defines no
    // section; the database's `[Req 3.1]` must be dangling, not accepted.
    const documents: DeepPlanDocuments = {
      requirements: "## *3.1 Authentication",
      database: "## 2.1 Users [Req 3.1]",
      tasks: "# Tasks",
    };

    const report = validateDocumentChain(documents);

    expect(report.ok).toBe(false);
    expect(report.issues).toEqual([
      { from: "database", ref: "Req 3.1", reason: "dangling" },
    ]);
  });

  it("reports requirements no task cites", () => {
    const documents = validChain();
    documents.tasks = "- [ ] T1 Login [Req 3.1]";

    const report = validateDocumentChain(documents);

    expect(report.uncovered).toEqual(["1", "3.1.1", "5.1"]);
    // Coverage gaps are reported but do not make the chain structurally invalid.
    expect(report.ok).toBe(true);
  });

  it("orders uncovered section numbers numerically, not lexically", () => {
    // `p10` is covered; `p2` and `p10.1` are not. A lexical sort would list
    // "10.1" before "2", so this pins the natural-number order.
    const documents: DeepPlanDocuments = {
      requirements: [
        "# Requirements",
        "",
        "## 2 Two",
        "",
        "## 10.1 Ten one",
        "",
        "## 10.2 Ten two",
      ].join("\n"),
      tasks: "# Tasks\n\n- [ ] T1 [Req 10.2]",
    };

    const report = validateDocumentChain(documents);

    expect(report.uncovered).toEqual(["2", "10.1"]);
  });

  it("stays silent about coverage when there is no tasks document yet", () => {
    const documents = validChain();
    delete documents.tasks;

    expect(validateDocumentChain(documents).uncovered).toEqual([]);
  });

  it("ignores documents past the phase it is asked to validate", () => {
    // Confirming `database` must not be blocked by a citation in `backend`,
    // which the user has not reached and cannot repair from that checkpoint.
    const documents = validChain();
    documents.backend = "## 4.2 Login [DB 9.9]";

    expect(validateDocumentChain(documents, "database").ok).toBe(true);
    expect(validateDocumentChain(documents).ok).toBe(false);
  });

  it("does not report coverage before the tasks phase is in range", () => {
    const documents = validChain();
    // Requirements define `1` and `3.1`; the tasks cite only `3.1`.
    documents.requirements = "## 1 Scope\n\n## 3.1 Authentication\n";
    documents.tasks = "- [ ] T1 Login [Req 3.1]";

    expect(validateDocumentChain(documents, "database").uncovered).toEqual([]);
    expect(validateDocumentChain(documents, "tasks").uncovered).toEqual(["1"]);
  });
});

describe("makeRefIssueMessage", () => {
  const t = ((key: string, options?: Record<string, unknown>) =>
    `${key}|${JSON.stringify(options ?? {})}`) as unknown as TFunction<"openhands">;

  it("names the producing document and the offending citation", () => {
    const message = makeRefIssueMessage(t, {
      from: "backend",
      ref: "DB 9.9.9",
      reason: "dangling",
    });
    expect(message).toContain(I18nKey.DEEP_PLAN$ISSUE_DANGLING);
    expect(message).toContain("backend-design.md");
    expect(message).toContain("DB 9.9.9");
  });

  it("falls back to the phase label for a phase without an output file", () => {
    const message = makeRefIssueMessage(t, {
      from: "analysis",
      ref: "DB 1.1",
      reason: "not-upstream",
    });
    expect(message).toContain(I18nKey.DEEP_PLAN$ISSUE_NOT_UPSTREAM);
    expect(message).toContain(I18nKey.DEEP_PLAN$PHASE_ANALYSIS);
  });
});
