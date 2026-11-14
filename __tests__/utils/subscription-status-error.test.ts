import { HttpError } from "@openhands/typescript-client";
import { describe, expect, it } from "vitest";
import { NoBackendAvailableError } from "#/api/agent-server-client-options";
import { OPENAI_SUBSCRIPTION_STATUS_PATH } from "#/constants/llm-subscription";
import { describeSubscriptionStatusError } from "#/utils/subscription-status-error";

describe("describeSubscriptionStatusError", () => {
  it("classifies a 404 as missing subscription endpoints", () => {
    // Arrange — an old agent-server without the subscription routes.
    const error = new HttpError(404, "Not Found", { detail: "Not Found" });

    // Act
    const info = describeSubscriptionStatusError(error);

    // Assert
    expect(info.kind).toBe("missing-endpoints");
    expect(info.detail).toContain(`GET ${OPENAI_SUBSCRIPTION_STATUS_PATH}`);
    expect(info.detail).toContain("HTTP 404");
  });

  it("classifies a 405 as missing subscription endpoints", () => {
    const error = new HttpError(405, "Method Not Allowed");

    expect(describeSubscriptionStatusError(error).kind).toBe(
      "missing-endpoints",
    );
  });

  it.each([401, 403])("classifies HTTP %i as unauthorized", (status) => {
    const error = new HttpError(status, "Denied");

    expect(describeSubscriptionStatusError(error).kind).toBe("unauthorized");
  });

  it("classifies a server error without exposing the response", () => {
    const error = new HttpError(500, "Internal Server Error", {
      detail: "PermissionError: chmod",
    });
    const info = describeSubscriptionStatusError(error);

    expect(info.kind).toBe("failed");
    expect(info.detail).toBe(
      `GET ${OPENAI_SUBSCRIPTION_STATUS_PATH} → HTTP 500`,
    );
  });

  it("distinguishes an unconfigured backend from a connection failure", () => {
    const info = describeSubscriptionStatusError(new NoBackendAvailableError());
    expect(info.kind).toBe("no-backend");
  });

  it("does not classify arbitrary errors as network failures", () => {
    const info = describeSubscriptionStatusError(
      new TypeError("Cannot read properties of null"),
    );
    expect(info.kind).toBe("failed");
  });

  it("recognizes raw Cloud fetch failures without exposing their messages", () => {
    const info = describeSubscriptionStatusError(
      new TypeError("Failed to fetch"),
    );
    expect(info.kind).toBe("unreachable");
    expect(info.detail).toBe(`GET ${OPENAI_SUBSCRIPTION_STATUS_PATH}`);
  });

  it("recognizes an SDK-wrapped fetch failure", () => {
    const error = new Error("Request failed: network failure", {
      cause: new TypeError("Failed to fetch"),
    });
    expect(describeSubscriptionStatusError(error).kind).toBe("unreachable");
  });

  it("does not mistake an SDK-wrapped non-network TypeError for a failed connection", () => {
    const error = new Error("Request failed", {
      cause: new TypeError("Cannot read properties of null"),
    });
    expect(describeSubscriptionStatusError(error).kind).toBe("failed");
  });

  it("does not expose server-controlled status text, response or exception messages", () => {
    const secret = "sensitive-example-token";
    const error = new HttpError(500, secret, { detail: secret });
    const detail = describeSubscriptionStatusError(error).detail;
    expect(detail).toBe(`GET ${OPENAI_SUBSCRIPTION_STATUS_PATH} → HTTP 500`);
    expect(detail).not.toContain(secret);
    expect(
      describeSubscriptionStatusError(new Error(secret)).detail,
    ).not.toContain(secret);
  });
});
