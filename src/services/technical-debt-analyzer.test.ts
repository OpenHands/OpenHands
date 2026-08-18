import { describe, it, expect } from "vitest";
import { TechnicalDebtAnalyzer, GENESIS_HASH } from "./technical-debt-analyzer";

describe("TechnicalDebtAnalyzer", () => {
  it("should score clean codebase with high production readiness", () => {
    const analyzer = new TechnicalDebtAnalyzer(75);
    const report = analyzer.analyzeCodebase("enterprise/ai-core", {
      unboundedLoops: 0,
      tokenMultiplier: 1.0,
      unGatedMutations: 0,
      cyclomaticComplexityAverage: 4.0,
    });

    expect(report.isProductionReady).toBe(true);
    expect(report.readinessScore).toBeGreaterThanOrEqual(90);
    expect(report.criticalSmells.length).toBe(0);
    expect(report.auditReceipt.currHash).toBeDefined();
  });

  it("should flag codebase with unbounded loops and un-gated mutations", () => {
    const analyzer = new TechnicalDebtAnalyzer(75);
    const report = analyzer.analyzeCodebase("legacy/hacky-agent-prototype", {
      unboundedLoops: 2,
      tokenMultiplier: 3.5,
      unGatedMutations: 3,
      cyclomaticComplexityAverage: 15.0,
    });

    expect(report.isProductionReady).toBe(false);
    expect(report.readinessScore).toBeLessThan(50);
    expect(report.criticalSmells).toContain(
      "DETECTED_2_UNBOUNDED_REASONING_LOOPS",
    );
    expect(report.criticalSmells).toContain(
      "DETECTED_3_UNGATED_PRODUCTION_MUTATIONS",
    );
  });

  it("should maintain cryptographic hash-chain integrity across multiple audits", () => {
    const analyzer = new TechnicalDebtAnalyzer();
    analyzer.analyzeCodebase("repo-alpha", { unboundedLoops: 0 });
    analyzer.analyzeCodebase("repo-beta", { unboundedLoops: 1 });
    analyzer.analyzeCodebase("repo-gamma", { unboundedLoops: 0 });

    const entries = analyzer.getLedger().getEntries();
    expect(entries.length).toBe(3);
    expect(entries[0].prevHash).toBe(GENESIS_HASH);
    expect(entries[1].prevHash).toBe(entries[0].currHash);
    expect(entries[2].prevHash).toBe(entries[1].currHash);
    expect(analyzer.getLedger().verifyIntegrity()).toBe(true);
  });
});
