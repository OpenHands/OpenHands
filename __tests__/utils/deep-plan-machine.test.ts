import { describe, expect, it } from "vitest";
import {
  canEnterPhase,
  confirmPhase,
  EMPTY_DEEP_PLAN_STATE,
  invalidateFrom,
  isPhaseConfirmed,
  startDeepPlan,
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
    expect(result.error).toContain("[Req 9.9]");
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

  it("refuses a phase whose predecessor is unconfirmed", () => {
    const state: DeepPlanState = {
      ...startDeepPlan(),
      activePhase: "database",
      confirmed: ["analysis"],
    };

    const result = confirmPhase(state, "database");

    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.error).toContain("requirements");
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

    expect(state.confirmed).toEqual([
      "analysis",
      "requirements",
      "database",
    ]);
    expect(state.activePhase).toBe("backend");
  });

  it("is a no-op for a phase that was never confirmed", () => {
    const state = invalidateFrom(confirmedChain, "tasks");

    expect(state).toBe(confirmedChain);
  });
});
