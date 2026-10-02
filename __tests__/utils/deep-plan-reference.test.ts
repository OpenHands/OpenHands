import { describe, expect, it } from "vitest";
import {
  describeRefIssue,
  extractDefinedSections,
  extractReferences,
  validateDocumentChain,
  type DeepPlanDocuments,
} from "#/utils/deep-plan-reference";

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
  backend: ["# Backend design", "", "## 4.2 Login endpoint [Req 3.1] [DB 2.5.1]"].join(
    "\n",
  ),
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

  it("reports requirements no task cites", () => {
    const documents = validChain();
    documents.tasks = "- [ ] T1 Login [Req 3.1]";

    const report = validateDocumentChain(documents);

    expect(report.uncovered).toEqual(["1", "3.1.1", "5.1"]);
    // Coverage gaps are reported but do not make the chain structurally invalid.
    expect(report.ok).toBe(true);
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

describe("describeRefIssue", () => {
  it("names the document and the offending citation", () => {
    expect(
      describeRefIssue({ from: "backend", ref: "DB 9.9.9", reason: "dangling" }),
    ).toContain("[DB 9.9.9]");
    expect(
      describeRefIssue({ from: "backend", ref: "DB 9.9.9", reason: "dangling" }),
    ).toContain("backend-design.md");
  });
});
