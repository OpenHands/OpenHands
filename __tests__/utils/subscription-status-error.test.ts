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

  it("classifies a server error as failed and carries the response detail", () => {
    // Arrange — a reachable agent-server whose status check raised.
    const error = new HttpError(500, "Internal Server Error", {
      detail: "PermissionError: chmod",
    });

    // Act
    const info = describeSubscriptionStatusError(error);

    // Assert
    expect(info.kind).toBe("failed");
    expect(info.detail).toContain("HTTP 500");
    expect(info.detail).toContain("PermissionError: chmod");
    expect(info.detail).toContain(`GET ${OPENAI_SUBSCRIPTION_STATUS_PATH}`);
  });

  it("classifies a transport failure as unreachable", () => {
    // Arrange — connection refused / offline, never reached the server.
    const error = new TypeError("fetch failed");

    // Act
    const info = describeSubscriptionStatusError(error);

    // Assert
    expect(info.kind).toBe("unreachable");
    expect(info.detail).toContain("fetch failed");
    expect(info.detail).toContain(`GET ${OPENAI_SUBSCRIPTION_STATUS_PATH}`);
  });

  it("classifies a missing backend as unreachable", () => {
    const info = describeSubscriptionStatusError(new NoBackendAvailableError());

    expect(info.kind).toBe("unreachable");
    expect(info.detail).toContain("No backend is configured.");
  });

  it("truncates very long details", () => {
    const error = new HttpError(500, "Internal Server Error", {
      detail: "x".repeat(500),
    });

    const info = describeSubscriptionStatusError(error);

    expect(info.detail.length).toBeLessThanOrEqual(200);
    expect(info.detail.endsWith("…")).toBe(true);
  });
});
