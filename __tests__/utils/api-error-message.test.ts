import { describe, expect, it } from "vitest";
import { AxiosError } from "axios";
import { HttpError } from "@openhands/typescript-client";
import {
  getApiErrorMessage,
  getApiOrConnectionErrorMessage,
} from "#/utils/api-error-message";
import {
  BACKEND_REQUEST_TIMEOUT_MESSAGE,
  CORS_OR_NETWORK_ERROR_MESSAGE,
} from "#/utils/user-facing-error";

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

  it("prefers a specific `detail` over a technical `exception`", () => {
    // Arrange — a friendly `detail` is the server's wording for users.
    const error = new HttpError(500, "Internal Server Error", {
      detail: "Plugin source could not be reached. Check the URL.",
      exception: "ConnectError: [Errno 111] Connection refused",
    });

    // Act + Assert
    expect(getApiErrorMessage(error, "fallback")).toBe(
      "Plugin source could not be reached. Check the URL.",
    );
  });

  it("prefers a validation `detail` array over `exception`", () => {
    const error = new HttpError(422, "Unprocessable Entity", {
      detail: [{ loc: ["body", "source"], msg: "Field required" }],
      exception: "RequestValidationError",
    });

    expect(getApiErrorMessage(error, "fallback")).toBe("Field required");
  });

  it.each([
    { state: "missing", body: {} },
    { state: "empty", body: { detail: "" } },
  ])("returns `exception` when `detail` is $state", ({ body }) => {
    const error = new HttpError(500, "Internal Server Error", {
      ...body,
      exception: "Local extension path does not exist: /plugins/magic-test",
    });

    expect(getApiErrorMessage(error, "fallback")).toBe(
      "Local extension path does not exist: /plugins/magic-test",
    );
  });

  it("returns the generic `detail` when there is no `exception`", () => {
    const error = new HttpError(500, "Internal Server Error", {
      detail: "Internal Server Error",
    });

    expect(getApiErrorMessage(error, "fallback")).toBe("Internal Server Error");
  });

  it("returns the `msg` of a single-entry FastAPI validation `detail` array", () => {
    // Arrange — a provider connection name over 128 characters (#17937).
    const error = new HttpError(422, "Unprocessable Entity", {
      detail: [
        {
          type: "string_too_long",
          loc: ["body", "display_name"],
          msg: "String should have at most 128 characters",
        },
      ],
    });

    // Act + Assert
    expect(getApiErrorMessage(error, "fallback")).toBe(
      "String should have at most 128 characters",
    );
  });

  it("joins the `msg` fields of a FastAPI validation `detail` array, naming each field", () => {
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
      "display_name: String should have at most 128 characters; provider: Field required",
    );
  });

  it("returns a string `error` field from an HttpError body", () => {
    // Arrange — some endpoints answer `{ error: "..." }` instead of `detail`.
    const error = new HttpError(
      400,
      "Bad Request",
      { error: "Repository is not connected" },
      'HTTP request failed (400 Bad Request): {"error":"Repository is not connected"}',
    );

    // Act + Assert
    expect(getApiErrorMessage(error, "fallback")).toBe(
      "Repository is not connected",
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

describe("getApiOrConnectionErrorMessage", () => {
  it("returns the server's `detail` for an HttpError", () => {
    const error = new HttpError(404, "Not Found", {
      detail: "Profile 'qa-nope' not found",
    });

    expect(getApiOrConnectionErrorMessage(error, "fallback")).toBe(
      "Profile 'qa-nope' not found",
    );
  });

  it("returns the fallback for an HttpError without a body", () => {
    const error = new HttpError(
      502,
      "Bad Gateway",
      null,
      "HTTP request failed (502 Bad Gateway): null",
    );

    expect(getApiOrConnectionErrorMessage(error, "fallback")).toBe("fallback");
  });

  it("keeps the shared disconnect wording for a network failure", () => {
    // The shared client wraps fetch failures in a plain Error with no body.
    const error = new Error("Request failed: Failed to fetch", {
      cause: new TypeError("Failed to fetch"),
    });

    expect(getApiOrConnectionErrorMessage(error, "fallback")).toBe(
      CORS_OR_NETWORK_ERROR_MESSAGE,
    );
  });

  it("keeps the shared timeout wording when the request times out", () => {
    const error = new Error("Request timeout after 60000ms");

    expect(getApiOrConnectionErrorMessage(error, "fallback")).toBe(
      BACKEND_REQUEST_TIMEOUT_MESSAGE,
    );
  });

  it("returns the fallback when the error carries no usable information", () => {
    expect(getApiOrConnectionErrorMessage(new Error(), "fallback")).toBe(
      "fallback",
    );
  });
});
