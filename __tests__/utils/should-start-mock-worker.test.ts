import { describe, expect, it } from "vitest";
import { shouldStartMockWorker } from "#/mocks/should-start-mock-worker";

describe("shouldStartMockWorker", () => {
  it("starts the worker whenever mock API mode is enabled in the browser", () => {
    expect(shouldStartMockWorker({ mockApi: "true", hasWindow: true })).toBe(
      true,
    );
  });

  it("does not start the worker when mock API mode is disabled", () => {
    expect(shouldStartMockWorker({ mockApi: "false", hasWindow: true })).toBe(
      false,
    );
    expect(shouldStartMockWorker({ mockApi: undefined, hasWindow: true })).toBe(
      false,
    );
  });

  it("does not start the worker during server-side rendering", () => {
    expect(shouldStartMockWorker({ mockApi: "true", hasWindow: false })).toBe(
      false,
    );
  });
});
