import { beforeEach, describe, expect, it } from "vitest";
import { useErrorMessageStore } from "#/stores/error-message-store";

const getState = () => useErrorMessageStore.getState();

describe("error message store", () => {
  beforeEach(() => {
    useErrorMessageStore.setState({
      errorMessage: null,
      errorType: null,
      errorCode: null,
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

    getState().setErrorMessage("auth failed", "conversation", "ACPAuthRequired");
    expect(getState().errorCode).toBe("ACPAuthRequired");
  });

  it("removeErrorMessage clears the error code too", () => {
    getState().setErrorMessage("auth failed", "conversation", "ACPAuthRequired");
    getState().removeErrorMessage();
    expect(getState().errorCode).toBeNull();
  });

  it("clearConnectionError clears the code for a connection error", () => {
    getState().setErrorMessage("offline", "connection", "SomeCode");
    getState().clearConnectionError();
    expect(getState().errorCode).toBeNull();
  });
});
