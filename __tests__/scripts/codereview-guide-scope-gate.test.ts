// @vitest-environment node
//
// Drift-detection for the repository-specific review guide
// (`.agents/skills/custom-codereview-guide.md`). Every PR to this repository must
// comply with this guide, and it owns the repository-ownership and
// maintainer-decision model. These assertions keep the scope gate as the first
// review step, expressed as stable categories rather than individual PR numbers,
// and keep the guide's front matter and cross-references valid.
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

const guidePath = ".agents/skills/custom-codereview-guide.md";

function read(rel: string): string {
  return readFileSync(path.join(repoRoot, rel), "utf-8");
}

const guide = read(guidePath);

describe("custom code-review guide scope gate", () => {
  it("keeps a valid skill name and trigger", () => {
    expect(guide.startsWith("---\n")).toBe(true);
    expect(guide).toContain("name: custom-codereview-guide");
    expect(guide).toContain("triggers:");
    expect(guide).toContain("/codereview");
  });

  it("is still referenced by AGENTS.md and docs/DEVELOPMENT.md", () => {
    expect(read("AGENTS.md")).toContain(guidePath);
    expect(read("docs/DEVELOPMENT.md")).toContain(
      `.agents/skills/custom-codereview-guide.md`,
    );
  });

  it("places the scope gate before the review sequence and blocking checkpoints", () => {
    const scopeGate = guide.indexOf("Scope Gate");
    expect(scopeGate).toBeGreaterThan(-1);
    expect(scopeGate).toBeLessThan(
      guide.indexOf("Review Sequence and Decision"),
    );
    expect(scopeGate).toBeLessThan(guide.indexOf("## Blocking Checkpoints"));
    expect(guide).toMatch(/before any detailed code inspection or tests/i);
  });

  it("defines the five stable scope categories", () => {
    const gates = [
      /wrong repository/i,
      /unconfirmed product or architecture direction/i,
      /obsolete or duplicate/i,
      /contrary to documented current direction/i,
      /clearly in-scope/i,
    ];

    for (const category of gates) {
      expect(guide).toMatch(category);
    }
  });

  it("routes wrong-repository work to the owning repository and a maintainer", () => {
    expect(guide).toMatch(
      /name the likely owning repository when the evidence is sufficient/i,
    );
    expect(guide).toMatch(/ask a maintainer to confirm/i);
  });

  it("requires a stated decision and maintainer confirmation for unconfirmed direction", () => {
    expect(guide).toMatch(/state the decision needed and ask a maintainer/i);
  });

  it("recommends closure with evidence for obsolete or contrary work", () => {
    expect(guide).toMatch(
      /recommend closure with evidence and ask a maintainer/i,
    );
  });

  it("keeps scope outcomes out of approval and exhaustive audits", () => {
    expect(guide).toMatch(/scope outcome never \*\*APPROVEs\*\* the PR/i);
    expect(guide).toMatch(
      /does not continue into an exhaustive implementation audit/i,
    );
  });

  it("routes clearly in-scope work through the existing checkpoints", () => {
    expect(guide).toMatch(
      /clearly in-scope changes proceed through the existing security,\n\s*correctness, architecture, evidence, and test checkpoints/i,
    );
  });

  it("does not enumerate individual PR numbers", () => {
    expect(guide).not.toMatch(/#\d{4,}/);
  });
});
