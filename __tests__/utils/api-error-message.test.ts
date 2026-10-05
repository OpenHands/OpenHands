import { describe, expect, it } from "vitest";
import { AxiosError } from "axios";
import { HttpError } from "@openhands/typescript-client";
import { getApiErrorMessage } from "#/utils/api-error-message";

describe("getApiErrorMessage", () => {
  it("returns the body `detail` from an HttpError when no `message` is present", () => {
    // Arrange — FastAPI-style error body on the shared client's HttpError.
    const error = new HttpError(422, "Unprocessable Entity", {
      detail: "Automation spec is invalid",
    });

    // Act + Assert
    expect(getApiErrorMessage(error, "fallback")).toBe(
      "Automation spec is invalid",
    );
  });

  it("returns the reason from an Agent Server unhandled-error body, not its generic `detail`", () => {
    // Arrange — the Agent Server answers unhandled errors with a fixed
    // `detail` and the actual reason under `exception`.
    const error = new HttpError(500, "Internal Server Error", {
      detail: "Internal Server Error",
      exception: "Local extension path does not exist: /plugins/magic-test",
      error_id: "0dd795f8",
    });

    // Act + Assert
    expect(getApiErrorMessage(error, "fallback")).toBe(
      "Local extension path does not exist: /plugins/magic-test",
    );
  });

  it("returns the response body `message` from an axios error", () => {
    // Arrange — local agent-server calls still reject with AxiosError.
    const error = new AxiosError("Request failed with status code 500");
    error.response = {
      status: 500,
      data: { message: "Runner exploded" },
    } as never;

    // Act + Assert
    expect(getApiErrorMessage(error, "fallback")).toBe("Runner exploded");
  });

  it("returns the fallback when the error carries no usable information", () => {
    expect(getApiErrorMessage(null, "fallback")).toBe("fallback");
  });
});
