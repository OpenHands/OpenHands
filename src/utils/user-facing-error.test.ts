import { describe, expect, it, vi } from "vitest";
import {
  BACKEND_REQUEST_TIMEOUT_MESSAGE,
  CORS_OR_NETWORK_ERROR_MESSAGE,
  getRawErrorMessage,
  getUserFacingConnectionErrorMessage,
  isBackendRequestTimeoutMessage,
  isCorsOrNetworkError,
  isCorsOrNetworkErrorMessage,
} from "./user-facing-error";

describe("user-facing connection errors", () => {
  it("maps browser fetch failures to the CORS or network message", () => {
    expect(
      getUserFacingConnectionErrorMessage(
        new Error("Request failed: Failed to fetch"),
      ),
    ).toBe(CORS_OR_NETWORK_ERROR_MESSAGE);
  });

  it("looks through wrapped causes from fetch-based clients", () => {
    expect(
      getUserFacingConnectionErrorMessage(
        new Error("Request failed", {
          cause: new TypeError("Failed to fetch"),
        }),
      ),
    ).toBe(CORS_OR_NETWORK_ERROR_MESSAGE);
  });

  it("detects existing CORS or network labels", () => {
    expect(isCorsOrNetworkErrorMessage(CORS_OR_NETWORK_ERROR_MESSAGE)).toBe(
      true,
    );
  });

  it("maps request timeouts to a backend timeout message", () => {
    expect(
      getUserFacingConnectionErrorMessage(
        new Error("Request timeout after 5000ms"),
      ),
    ).toBe(BACKEND_REQUEST_TIMEOUT_MESSAGE);
  });

  it("leaves ordinary server errors intact", () => {
    expect(
      getUserFacingConnectionErrorMessage(new Error("Invalid API key")),
    ).toBe("Invalid API key");
  });

  it.each([
    "Disconnected (check CORS or network)",
    "Disconnected (check URL or network)",
    "blocked by CORS",
    "failed to fetch",
    "network error",
    "load failed",
    "NetworkError when attempting to fetch resource",
    "CORS policy blocked this request",
  ])("recognizes network failure wording: %s", (message) => {
    expect(isCorsOrNetworkErrorMessage(message)).toBe(true);
  });

  it("rejects absent and ordinary messages as network failures", () => {
    expect(isCorsOrNetworkErrorMessage(undefined)).toBe(false);
    expect(isCorsOrNetworkErrorMessage("ordinary failure")).toBe(false);
  });

  it.each(["CORS configuration issue", "request blocked by policy"])(
    "does not classify incomplete network wording: %s",
    (message) => expect(isCorsOrNetworkErrorMessage(message)).toBe(false),
  );

  it.each([
    "request timeout",
    "timeout after 10s",
    "backend request timed out",
  ])("recognizes backend timeout wording: %s", (message) =>
    expect(isBackendRequestTimeoutMessage(message)).toBe(true),
  );

  it("rejects absent and ordinary messages as timeouts", () => {
    expect(isBackendRequestTimeoutMessage(null)).toBe(false);
    expect(isBackendRequestTimeoutMessage("ordinary failure")).toBe(false);
  });

  it("collects messages from strings and record-shaped causes", () => {
    const wrapped = {
      message: "outer",
      cause: { message: "Failed to fetch" },
    };
    expect(getRawErrorMessage(wrapped)).toBe("outer");
    expect(isCorsOrNetworkError(wrapped)).toBe(true);
    expect(getRawErrorMessage("plain string")).toBe("plain string");
    expect(getRawErrorMessage({ message: 42, cause: "nested" })).toBe("nested");
  });

  it("ignores callable values with error-like properties", () => {
    const callable = Object.assign(() => undefined, {
      message: "Failed to fetch",
    });

    expect(getRawErrorMessage(callable)).toBeNull();
  });

  it("handles empty, primitive, cyclic, and deeply nested causes", () => {
    expect(getRawErrorMessage(undefined)).toBeNull();
    expect(getRawErrorMessage(42)).toBeNull();
    expect(getRawErrorMessage("")).toBeNull();
    expect(getRawErrorMessage(new Error(""))).toBeNull();
    expect(getRawErrorMessage({ message: "" })).toBeNull();

    const cyclic: { message: string; cause?: unknown } = { message: "cycle" };
    cyclic.cause = cyclic;
    expect(getRawErrorMessage(cyclic)).toBe("cycle");

    const tooDeep = {
      message: "one",
      cause: {
        message: "two",
        cause: {
          message: "three",
          cause: {
            message: "four",
            cause: { message: "Failed to fetch" },
          },
        },
      },
    };
    expect(isCorsOrNetworkError(tooDeep)).toBe(false);
  });

  it("stops collecting when a cyclic cause is encountered again", () => {
    let messageReads = 0;
    const cyclic: { readonly message: string; cause?: unknown } = {
      get message() {
        messageReads += 1;
        return messageReads === 1 ? "ordinary failure" : "Failed to fetch";
      },
    };
    cyclic.cause = cyclic;

    expect(getUserFacingConnectionErrorMessage(cyclic)).toBe(
      "ordinary failure",
    );
  });

  it("returns null when no user-facing message can be extracted", () => {
    expect(getUserFacingConnectionErrorMessage({})).toBeNull();
  });

  it("exports actionable connection guidance on module initialization", async () => {
    vi.resetModules();

    try {
      const fresh = await import("./user-facing-error");

      expect(fresh.CORS_OR_NETWORK_ERROR_MESSAGE).toContain(
        "check URL or network",
      );
      expect(fresh.BACKEND_REQUEST_TIMEOUT_MESSAGE).toContain(
        "request timed out",
      );
    } finally {
      vi.resetModules();
    }
  });
});
