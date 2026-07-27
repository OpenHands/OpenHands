import { describe, expect, it } from "vitest";
import {
  BACKEND_REQUEST_TIMEOUT_MESSAGE,
  CORS_OR_NETWORK_ERROR_MESSAGE,
  getUserFacingConnectionErrorMessage,
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
});
