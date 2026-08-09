import { beforeEach, describe, expect, it } from "vitest";
import {
  SECURITY_SCAN_RESULTS_STORAGE_KEY,
  useSecurityScanResultsStore,
} from "#/stores/security-scan-results-store";
import type { ScaScanResult, SecurityScanResult } from "#/types/security-scan";

const CONVERSATION_A = "conversation-a";
const CONVERSATION_B = "conversation-b";

const sastResult: SecurityScanResult = {
  tool: "opengrep",
  scannedAt: "2026-08-09T12:00:00.000Z",
  findings: [
    {
      id: "f1",
      ruleId: "rule-1",
      message: "Issue",
      severity: "HIGH",
      filePath: "src/a.ts",
      startLine: 1,
      startCol: 1,
      endLine: 1,
      endCol: 2,
    },
  ],
};

const scaResult: ScaScanResult = {
  tool: "dependency-track",
  scannedAt: "2026-08-09T12:05:00.000Z",
  findings: [
    {
      id: "c1",
      packageName: "left-pad",
      packageVersion: "1.0.0",
      purl: "pkg:npm/left-pad@1.0.0",
      cveId: "CVE-2026-0001",
      severity: "CRITICAL",
      description: "Bad",
    },
  ],
};

describe("security-scan-results store", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useSecurityScanResultsStore.setState({ resultsByConversationId: {} });
  });

  it("stores SAST and SCA results per conversation without mixing them", () => {
    const store = useSecurityScanResultsStore.getState();
    store.setSastResult(CONVERSATION_A, sastResult);
    store.setScaResult(CONVERSATION_A, scaResult);
    store.setSastResult(CONVERSATION_B, {
      ...sastResult,
      scannedAt: "2026-08-09T13:00:00.000Z",
      findings: [],
    });

    const state = useSecurityScanResultsStore.getState();
    expect(state.resultsByConversationId[CONVERSATION_A]).toEqual({
      sast: sastResult,
      sca: scaResult,
    });
    expect(state.resultsByConversationId[CONVERSATION_B]?.sast?.findings).toEqual(
      [],
    );
    expect(state.resultsByConversationId[CONVERSATION_B]?.sca).toBeNull();
  });

  it("clears one conversation without touching others", () => {
    const store = useSecurityScanResultsStore.getState();
    store.setSastResult(CONVERSATION_A, sastResult);
    store.setSastResult(CONVERSATION_B, sastResult);

    store.clearConversation(CONVERSATION_A);

    const state = useSecurityScanResultsStore.getState();
    expect(state.resultsByConversationId[CONVERSATION_A]).toBeUndefined();
    expect(state.resultsByConversationId[CONVERSATION_B]?.sast).toEqual(
      sastResult,
    );
  });

  it("persists results to localStorage so remounts can restore them", () => {
    useSecurityScanResultsStore
      .getState()
      .setSastResult(CONVERSATION_A, sastResult);

    const raw = window.localStorage.getItem(SECURITY_SCAN_RESULTS_STORAGE_KEY);
    expect(raw).toBeTruthy();
    expect(raw).toContain(CONVERSATION_A);
    expect(raw).toContain("opengrep");
  });
});
