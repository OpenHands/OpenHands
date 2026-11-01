import { describe, expect, it } from "vitest";
import { AxiosError, AxiosHeaders } from "axios";
import { HttpError } from "@openhands/typescript-client";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import {
  BACKEND_REQUEST_TIMEOUT_MESSAGE,
  CORS_OR_NETWORK_ERROR_MESSAGE,
} from "#/utils/user-facing-error";

// The shared TypeScript client throws an HttpError whose message is the raw
// transport text and whose parsed body is on `response`.
function createHttpError(status: number, statusText: string, body: unknown) {
  return new HttpError(
    status,
    statusText,
    body,
    `HTTP request failed (${status} ${statusText}): ${JSON.stringify(body)}`,
  );
}

function createAxiosErrorWithBody(body: unknown) {
  const error = new AxiosError("Request failed with status code 400");
  error.response = {
    data: body,
    status: 400,
    statusText: "Bad Request",
    headers: {},
    config: { headers: new AxiosHeaders() },
  };
  return error;
}

describe("retrieveAxiosErrorMessage", () => {
  it.each([
    {
      description: "a string `detail`",
      error: createHttpError(422, "Unprocessable Entity", {
        detail: "Settings validation failed",
      }),
      expected: "Settings validation failed",
    },
    {
      description: "a FastAPI validation `detail` array",
      error: createHttpError(422, "Unprocessable Entity", {
        detail: [
          {
            type: "string_too_long",
            loc: ["body", "display_name"],
            msg: "String should have at most 128 characters",
          },
        ],
      }),
      expected: "String should have at most 128 characters",
    },
    {
      description: "an `error` field",
      error: createHttpError(400, "Bad Request", {
        error: "Repository is not connected",
      }),
      expected: "Repository is not connected",
    },
  ])(
    "returns the server's message from an SDK HttpError with $description",
    ({ error, expected }) => {
      expect(retrieveAxiosErrorMessage(error)).toBe(expected);
    },
  );

  it.each([
    {
      description: "an object without a message field",
      body: { unexpected: true },
    },
    {
      description: "a plain-text gateway body",
      body: "Bad Gateway: connect ECONNREFUSED 127.0.0.1:8001",
    },
    { description: "no body", body: null },
  ])(
    "returns an empty string, never the raw transport text, for an SDK HttpError with $description",
    ({ body }) => {
      // Callers then show their own localized fallback.
      expect(
        retrieveAxiosErrorMessage(createHttpError(502, "Bad Gateway", body)),
      ).toBe("");
    },
  );

  it.each([
    {
      description: "an axios error's `error` field",
      error: createAxiosErrorWithBody({ error: "Axios error field" }),
      expected: "Axios error field",
    },
    {
      description: "an axios error's `message` field",
      error: createAxiosErrorWithBody({ message: "Axios message field" }),
      expected: "Axios message field",
    },
    {
      description: "an axios error's own message when the body has neither",
      error: createAxiosErrorWithBody({ unexpected: true }),
      expected: "Request failed with status code 400",
    },
    {
      description: "the disconnect wording for an axios network error",
      error: new AxiosError("Network Error"),
      expected: CORS_OR_NETWORK_ERROR_MESSAGE,
    },
    {
      description: "a plain Error's message",
      error: new Error("Something specific broke"),
      expected: "Something specific broke",
    },
    {
      description: "the disconnect wording for a fetch failure",
      error: new Error("Request failed: Failed to fetch", {
        cause: new TypeError("Failed to fetch"),
      }),
      expected: CORS_OR_NETWORK_ERROR_MESSAGE,
    },
    {
      description: "the timeout wording for a timed-out request",
      error: new Error("Request timeout after 60000ms"),
      expected: BACKEND_REQUEST_TIMEOUT_MESSAGE,
    },
    {
      description: "a string error as is",
      error: "Plain string failure",
      expected: "Plain string failure",
    },
    {
      description: "an empty string when nothing is usable",
      error: {},
      expected: "",
    },
  ])("keeps returning $description", ({ error, expected }) => {
    expect(retrieveAxiosErrorMessage(error)).toBe(expected);
  });
});
