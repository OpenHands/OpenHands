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

  it("joins the `msg` fields of a FastAPI validation `detail` array", () => {
    // Arrange — Pydantic 422 bodies carry `detail` as a list of errors.
    const error = new HttpError(422, "Unprocessable Entity", {
      detail: [
        {
          type: "string_too_long",
          loc: ["body", "display_name"],
          msg: "String should have at most 128 characters",
        },
        { type: "missing", loc: ["body", "provider"], msg: "Field required" },
      ],
    });

    // Act + Assert
    expect(getApiErrorMessage(error, "fallback")).toBe(
      "String should have at most 128 characters; Field required",
    );
  });

  it("returns the fallback instead of the raw transport text for an HttpError without a usable body", () => {
    // Arrange — the client's message is `HTTP request failed (...): <json>`.
    const error = new HttpError(
      500,
      "Internal Server Error",
      { unexpected: true },
      'HTTP request failed (500 Internal Server Error): {"unexpected":true}',
    );

    // Act + Assert
    expect(getApiErrorMessage(error, "fallback")).toBe("fallback");
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
