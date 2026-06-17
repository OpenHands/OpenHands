import { describe, expect, it } from "vitest";
import { AxiosError, AxiosHeaders } from "axios";
import {
  isConcurrencyLimitError,
  getConcurrencyLimit,
} from "#/utils/concurrency-limit-error";
import { DEFAULT_CONCURRENT_SANDBOX_LIMIT } from "#/utils/constants";

describe("isConcurrencyLimitError", () => {
  it("returns true for valid 429 concurrency limit error", () => {
    // FastAPI wraps HTTPException detail in a "detail" field
    const error = new AxiosError("Too Many Requests", "429", undefined, null, {
      status: 429,
      statusText: "Too Many Requests",
      headers: new AxiosHeaders(),
      config: { headers: new AxiosHeaders() },
      data: {
        detail: {
          error: "CONCURRENCY_LIMIT_REACHED",
          message: "You have reached your limit",
          limit: 3,
          current: 3,
        },
      },
    });

    expect(isConcurrencyLimitError(error)).toBe(true);
  });

  it("returns false for non-AxiosError", () => {
    const error = new Error("Some error");
    expect(isConcurrencyLimitError(error)).toBe(false);
  });

  it("returns false for null", () => {
    expect(isConcurrencyLimitError(null)).toBe(false);
  });

  it("returns false for undefined", () => {
    expect(isConcurrencyLimitError(undefined)).toBe(false);
  });

  it("returns false for string error", () => {
    expect(isConcurrencyLimitError("some error")).toBe(false);
  });

  it("returns false for non-429 status code", () => {
    const error = new AxiosError("Server Error", "500", undefined, null, {
      status: 500,
      statusText: "Internal Server Error",
      headers: new AxiosHeaders(),
      config: { headers: new AxiosHeaders() },
      data: {
        detail: {
          error: "CONCURRENCY_LIMIT_REACHED",
          message: "You have reached your limit",
          limit: 3,
          current: 3,
        },
      },
    });

    expect(isConcurrencyLimitError(error)).toBe(false);
  });

  it("returns false for 429 with different error code", () => {
    const error = new AxiosError("Too Many Requests", "429", undefined, null, {
      status: 429,
      statusText: "Too Many Requests",
      headers: new AxiosHeaders(),
      config: { headers: new AxiosHeaders() },
      data: {
        detail: {
          error: "RATE_LIMITED",
          message: "Too many requests",
        },
      },
    });

    expect(isConcurrencyLimitError(error)).toBe(false);
  });

  it("returns false for 429 with no error code", () => {
    const error = new AxiosError("Too Many Requests", "429", undefined, null, {
      status: 429,
      statusText: "Too Many Requests",
      headers: new AxiosHeaders(),
      config: { headers: new AxiosHeaders() },
      data: {
        detail: {
          message: "Too many requests",
        },
      },
    });

    expect(isConcurrencyLimitError(error)).toBe(false);
  });

  it("returns false for 429 with no response data", () => {
    const error = new AxiosError("Too Many Requests", "429", undefined, null, {
      status: 429,
      statusText: "Too Many Requests",
      headers: new AxiosHeaders(),
      config: { headers: new AxiosHeaders() },
      data: null,
    });

    expect(isConcurrencyLimitError(error)).toBe(false);
  });

  it("returns false for AxiosError with no response", () => {
    const error = new AxiosError("Network Error", "ERR_NETWORK");
    expect(isConcurrencyLimitError(error)).toBe(false);
  });
});

describe("getConcurrencyLimit", () => {
  it("extracts limit from error response", () => {
    // FastAPI wraps HTTPException detail in a "detail" field
    const error = new AxiosError("Too Many Requests", "429", undefined, null, {
      status: 429,
      statusText: "Too Many Requests",
      headers: new AxiosHeaders(),
      config: { headers: new AxiosHeaders() },
      data: {
        detail: {
          error: "CONCURRENCY_LIMIT_REACHED" as const,
          message: "You have reached your limit",
          limit: 5,
          current: 5,
        },
      },
    });

    expect(getConcurrencyLimit(error)).toBe(5);
  });

  it("returns default limit when limit is missing", () => {
    const error = new AxiosError("Too Many Requests", "429", undefined, null, {
      status: 429,
      statusText: "Too Many Requests",
      headers: new AxiosHeaders(),
      config: { headers: new AxiosHeaders() },
      data: {
        detail: {
          error: "CONCURRENCY_LIMIT_REACHED" as const,
          message: "You have reached your limit",
          limit: undefined as unknown as number,
          current: 3,
        },
      },
    });

    expect(getConcurrencyLimit(error)).toBe(DEFAULT_CONCURRENT_SANDBOX_LIMIT);
  });

  it("returns default limit when response data is null", () => {
    const error = new AxiosError("Too Many Requests", "429", undefined, null, {
      status: 429,
      statusText: "Too Many Requests",
      headers: new AxiosHeaders(),
      config: { headers: new AxiosHeaders() },
      data: null,
    });

    // @ts-expect-error - testing with invalid data type
    expect(getConcurrencyLimit(error)).toBe(DEFAULT_CONCURRENT_SANDBOX_LIMIT);
  });

  it("returns default limit when response is undefined", () => {
    const error = new AxiosError("Network Error", "ERR_NETWORK");
    // @ts-expect-error - testing with no response
    expect(getConcurrencyLimit(error)).toBe(DEFAULT_CONCURRENT_SANDBOX_LIMIT);
  });

  it("extracts different limit values correctly", () => {
    // FastAPI wraps HTTPException detail in a "detail" field
    const createError = (limit: number) =>
      new AxiosError("Too Many Requests", "429", undefined, null, {
        status: 429,
        statusText: "Too Many Requests",
        headers: new AxiosHeaders(),
        config: { headers: new AxiosHeaders() },
        data: {
          detail: {
            error: "CONCURRENCY_LIMIT_REACHED" as const,
            message: "You have reached your limit",
            limit,
            current: limit,
          },
        },
      });

    expect(getConcurrencyLimit(createError(1))).toBe(1);
    expect(getConcurrencyLimit(createError(10))).toBe(10);
    expect(getConcurrencyLimit(createError(100))).toBe(100);
  });
});
