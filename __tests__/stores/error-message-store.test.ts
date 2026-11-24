import { beforeEach, describe, expect, it } from "vitest";
import { useErrorMessageStore } from "#/stores/error-message-store";

const getState = () => useErrorMessageStore.getState();

describe("error message store", () => {
  beforeEach(() => {
    useErrorMessageStore.setState({
      errorMessage: null,
      errorType: null,
      errorCode: null,
      errorClassification: null,
      plannerError: null,
    });
  });

  it("defaults to a sticky conversation error", () => {
    getState().setErrorMessage("boom");
    expect(getState().errorMessage).toBe("boom");
    expect(getState().errorType).toBe("conversation");
  });

  it("tags connection errors when the type is provided", () => {
    getState().setErrorMessage("offline", "connection");
    expect(getState().errorType).toBe("connection");
  });

  it("removeErrorMessage clears any error regardless of type", () => {
    getState().setErrorMessage("boom");
    getState().removeErrorMessage();
    expect(getState().errorMessage).toBeNull();
    expect(getState().errorType).toBeNull();
  });

  it("clearConnectionError clears a transient connection error", () => {
    getState().setErrorMessage("offline", "connection");
    getState().clearConnectionError();
    expect(getState().errorMessage).toBeNull();
    expect(getState().errorType).toBeNull();
  });

  it("clearConnectionError preserves a sticky conversation error", () => {
    getState().setErrorMessage("bad api key");
    getState().clearConnectionError();
    expect(getState().errorMessage).toBe("bad api key");
    expect(getState().errorType).toBe("conversation");
  });

  it("clearConnectionError is a no-op when there is no error", () => {
    getState().clearConnectionError();
    expect(getState().errorMessage).toBeNull();
    expect(getState().errorType).toBeNull();
  });

  it("stores the optional error code and defaults it to null", () => {
    getState().setErrorMessage("boom");
    expect(getState().errorCode).toBeNull();

    getState().setErrorMessage(
      "auth failed",
      "conversation",
      "ACPAuthRequired",
    );
    expect(getState().errorCode).toBe("ACPAuthRequired");
  });

  it("stores and clears structured error classifications", () => {
    const classification = {
      kind: "auth" as const,
      retryable: false,
      user_action: "settings" as const,
    };

    getState().setErrorMessage(
      "auth failed",
      "conversation",
      null,
      classification,
    );
    expect(getState().errorClassification).toEqual(classification);

    getState().removeErrorMessage();
    expect(getState().errorClassification).toBeNull();
  });

  it("removeErrorMessage clears the error code too", () => {
    getState().setErrorMessage(
      "auth failed",
      "conversation",
      "ACPAuthRequired",
    );
    getState().removeErrorMessage();
    expect(getState().errorCode).toBeNull();
  });

  it("clearConnectionError clears the code for a connection error", () => {
    getState().setErrorMessage("offline", "connection", "SomeCode");
    getState().clearConnectionError();
    expect(getState().errorCode).toBeNull();
  });

  describe("planner errors", () => {
    it("keeps main and planner errors apart in either order", () => {
      getState().setErrorMessage("main failed", "conversation", "MainCode");
      getState().setPlannerError("planner failed", "conversation", "PlanCode");
      getState().setErrorMessage("main failed again");

      expect(getState().errorMessage).toBe("main failed again");
      expect(getState().plannerError).toEqual({
        message: "planner failed",
        type: "conversation",
        code: "PlanCode",
        classification: null,
      });
    });

    it("removes one role's error without touching the other", () => {
      getState().setErrorMessage("main failed");
      getState().setPlannerError("planner failed");

      getState().removePlannerError();
      expect(getState().plannerError).toBeNull();
      expect(getState().errorMessage).toBe("main failed");

      getState().setPlannerError("planner failed");
      getState().removeErrorMessage();
      expect(getState().errorMessage).toBeNull();
      expect(getState().plannerError?.message).toBe("planner failed");
    });

    it("clears connection errors per role", () => {
      getState().setErrorMessage("main offline", "connection");
      getState().setPlannerError("planner offline", "connection");

      getState().clearPlannerConnectionError();
      expect(getState().plannerError).toBeNull();
      expect(getState().errorMessage).toBe("main offline");

      getState().setPlannerError("planner offline", "connection");
      getState().clearConnectionError();
      expect(getState().errorMessage).toBeNull();
      expect(getState().plannerError?.message).toBe("planner offline");
    });

    it("clearPlannerConnectionError preserves a sticky planner error", () => {
      getState().setPlannerError("bad api key");
      getState().clearPlannerConnectionError();
      expect(getState().plannerError?.message).toBe("bad api key");
    });

    it("resetErrorMessages clears both roles", () => {
      getState().setErrorMessage("main failed");
      getState().setPlannerError("planner failed");

      getState().resetErrorMessages();

      expect(getState().errorMessage).toBeNull();
      expect(getState().plannerError).toBeNull();
    });
  });
});
