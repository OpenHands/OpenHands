import { describe, expect, it } from "vitest";
import {
  canEnterPhase,
  confirmPhase,
  EMPTY_DEEP_PLAN_STATE,
  hashDeepPlanDocument,
  hydrateDeepPlanState,
  invalidateFrom,
  isPhaseConfirmed,
  startDeepPlan,
  toPersistedDeepPlan,
  type DeepPlanState,
} from "#/utils/deep-plan-machine";

const requirements = ["# Requirements", "", "## 3.1 Authentication"].join("\n");
const database = ["# Database design", "", "## 2.1 Users [Req 3.1]"].join("\n");

describe("startDeepPlan", () => {
  it("opens on the first phase with nothing confirmed", () => {
    const state = startDeepPlan();

    expect(state.activePhase).toBe("analysis");
    expect(state.confirmed).toEqual([]);
  });
});

describe("canEnterPhase", () => {
  it("always allows the first phase", () => {
    expect(canEnterPhase(EMPTY_DEEP_PLAN_STATE, "analysis")).toBe(true);
  });

  it("blocks a phase until every earlier phase is confirmed", () => {
    const state: DeepPlanState = {
      ...EMPTY_DEEP_PLAN_STATE,
      confirmed: ["analysis"],
    };

    expect(canEnterPhase(state, "requirements")).toBe(true);
    expect(canEnterPhase(state, "database")).toBe(false);
  });

  it("keeps a confirmed phase enterable so it can be revised", () => {
    const state: DeepPlanState = {
      ...EMPTY_DEEP_PLAN_STATE,
      confirmed: ["analysis", "requirements"],
    };

    expect(canEnterPhase(state, "requirements")).toBe(true);
  });
});

describe("confirmPhase", () => {
  it("advances to the next phase once the checkpoint passes", () => {
    const result = confirmPhase(startDeepPlan(), "analysis");

    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.state.activePhase).toBe("requirements");
    expect(isPhaseConfirmed(result.state, "analysis")).toBe(true);
  });

  it("blocks the transition while the reference chain is invalid", () => {
    const state: DeepPlanState = {
      ...startDeepPlan(),
      activePhase: "database",
      confirmed: ["analysis", "requirements"],
      documents: {
        requirements,
        // [Req 9.9] does not exist upstream — the validator must block this.
        database: "## 2.1 Users [Req 9.9]",
      },
    };

    const result = confirmPhase(state, "database");

    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.failure).toEqual({
      kind: "invalid-chain",
      issue: { from: "database", ref: "Req 9.9", reason: "dangling" },
      extraCount: 0,
    });
  });

  it("does not let a stale downstream citation block an upstream phase", () => {
    // The backend document still cites a section the database document has
    // since dropped. The user is confirming `database` and cannot repair
    // `backend` from that checkpoint, so it must not gate the transition.
    const state: DeepPlanState = {
      ...startDeepPlan(),
      activePhase: "database",
      confirmed: ["analysis", "requirements"],
      documents: {
        requirements,
        database: "## 2.1 Users [Req 3.1]",
        backend: "## 4.2 Login [DB 9.9]",
      },
    };

    const result = confirmPhase(state, "database");

    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(isPhaseConfirmed(result.state, "database")).toBe(true);
  });

  it("refuses a phase that must produce a document when none exists yet", () => {
    // `requirements` writes `requirements.md`; confirming it before the
    // planner has produced the file would advance the chain on no evidence.
    const state: DeepPlanState = {
      ...startDeepPlan(),
      activePhase: "requirements",
      confirmed: ["analysis"],
      documents: {},
    };

    const result = confirmPhase(state, "requirements");

    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.failure).toEqual({
      kind: "missing-output",
      phase: "requirements",
    });
  });

  it("confirms a document-producing phase once its document exists", () => {
    const state: DeepPlanState = {
      ...startDeepPlan(),
      activePhase: "requirements",
      confirmed: ["analysis"],
      documents: { requirements },
    };

    const result = confirmPhase(state, "requirements");

    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(isPhaseConfirmed(result.state, "requirements")).toBe(true);
    expect(result.state.activePhase).toBe("database");
  });

  it("confirms a pure-conversation phase without any document", () => {
    // `analysis` and `implementation` have no output file, so the missing-output
    // gate must not apply to them.
    const analysis = confirmPhase(startDeepPlan(), "analysis");

    expect(analysis.ok).toBe(true);
    if (!analysis.ok) return;
    expect(isPhaseConfirmed(analysis.state, "analysis")).toBe(true);
  });

  it("refuses a phase whose predecessor is unconfirmed", () => {
    const state: DeepPlanState = {
      ...startDeepPlan(),
      activePhase: "database",
      confirmed: ["analysis"],
    };

    const result = confirmPhase(state, "database");

    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.failure).toEqual({ kind: "blocked", phase: "requirements" });
  });

  it("does not duplicate an already-confirmed phase", () => {
    const first = confirmPhase(startDeepPlan(), "analysis");
    expect(first.ok).toBe(true);
    if (!first.ok) return;

    const again = confirmPhase(first.state, "analysis");
    expect(again.ok).toBe(true);
    if (!again.ok) return;
    expect(again.state.confirmed).toEqual(["analysis"]);
  });

  it("accepts a chain whose citations resolve upstream", () => {
    const state: DeepPlanState = {
      ...startDeepPlan(),
      activePhase: "database",
      confirmed: ["analysis", "requirements"],
      documents: { requirements, database },
    };

    const result = confirmPhase(state, "database");

    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.state.activePhase).toBe("backend");
  });

  it("resolves a citation to a punctuation-separated heading", () => {
    // `## 3.1: Authentication` still defines section 3.1; the checkpoint must
    // resolve `[Req 3.1]` rather than block with a dangling reference.
    const state: DeepPlanState = {
      ...startDeepPlan(),
      activePhase: "database",
      confirmed: ["analysis", "requirements"],
      documents: {
        requirements: ["# Requirements", "", "## 3.1: Authentication"].join(
          "\n",
        ),
        database,
      },
    };

    const result = confirmPhase(state, "database");

    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.state.activePhase).toBe("backend");
  });

  it("blocks a citation that resolves only to a numeric quantity", () => {
    // `## 3,000 concurrent users` is a quantity, not section 3; the database
    // checkpoint must refuse rather than advance.
    const state: DeepPlanState = {
      ...startDeepPlan(),
      activePhase: "database",
      confirmed: ["analysis", "requirements"],
      documents: {
        requirements: ["# Requirements", "", "## 3,000 concurrent users"].join(
          "\n",
        ),
        database,
      },
    };

    const result = confirmPhase(state, "database");

    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.failure).toEqual({
      kind: "invalid-chain",
      issue: { from: "database", ref: "Req 3.1", reason: "dangling" },
      extraCount: 0,
    });
  });
});

describe("invalidateFrom", () => {
  const confirmedChain: DeepPlanState = {
    activePhase: "tasks",
    confirmed: ["analysis", "requirements", "database", "backend"],
    documents: { requirements, database },
  };

  it("drops the rewritten phase and every later confirmation", () => {
    const state = invalidateFrom(confirmedChain, "requirements");

    expect(state.confirmed).toEqual(["analysis"]);
  });

  it("pulls the active phase back to the rewritten one", () => {
    // The chain can no longer be trusted past the edit, so continuing at
    // `tasks` would build on unconfirmed documents.
    const state = invalidateFrom(confirmedChain, "requirements");

    expect(state.activePhase).toBe("requirements");
  });

  it("keeps earlier confirmations and the active phase when a later doc is edited", () => {
    const state = invalidateFrom(confirmedChain, "backend");

    expect(state.confirmed).toEqual(["analysis", "requirements", "database"]);
    expect(state.activePhase).toBe("backend");
  });

  it("is a no-op for a phase that was never confirmed", () => {
    const state = invalidateFrom(confirmedChain, "tasks");

    expect(state).toBe(confirmedChain);
  });
});

describe("persisted deep-plan shape", () => {
  it("drops document bodies but keeps the phase, confirmations and hashes", () => {
    const state: DeepPlanState = {
      ...startDeepPlan(),
      activePhase: "database",
      confirmed: ["analysis", "requirements"],
      documents: { requirements },
      documentHashes: { requirements: hashDeepPlanDocument(requirements) },
    };

    const persisted = toPersistedDeepPlan(state);

    expect(persisted).toEqual({
      activePhase: "database",
      confirmed: ["analysis", "requirements"],
      documentHashes: { requirements: hashDeepPlanDocument(requirements) },
    });
    expect(persisted).not.toHaveProperty("documents");
  });

  it("rebuilds the machine with empty bodies from a persisted blob", () => {
    const state: DeepPlanState = {
      ...startDeepPlan(),
      activePhase: "database",
      confirmed: ["analysis", "requirements"],
      documents: { requirements },
      documentHashes: { requirements: hashDeepPlanDocument(requirements) },
    };

    const restored = hydrateDeepPlanState(toPersistedDeepPlan(state));

    expect(restored.activePhase).toBe("database");
    expect(restored.confirmed).toEqual(["analysis", "requirements"]);
    expect(restored.documents).toEqual({});
    expect(restored.documentHashes).toEqual({
      requirements: hashDeepPlanDocument(requirements),
    });
  });

  it("returns the empty machine for a missing persisted blob", () => {
    expect(hydrateDeepPlanState(undefined)).toEqual(EMPTY_DEEP_PLAN_STATE);
  });
});
