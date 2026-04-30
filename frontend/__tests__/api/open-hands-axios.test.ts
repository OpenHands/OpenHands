import { afterEach, describe, expect, it, vi } from "vitest";
import { resolveBackendBaseURL } from "#/api/open-hands-axios";

describe("resolveBackendBaseURL", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("uses the current window origin when env is unset", () => {
    vi.stubEnv("VITE_BACKEND_BASE_URL", "");

    expect(resolveBackendBaseURL()).toBe(
      `${window.location.protocol}//${window.location.host}`,
    );
  });

  it("prepends the current protocol for host and port env values", () => {
    vi.stubEnv("VITE_BACKEND_BASE_URL", "api.example.test:3000/");

    expect(resolveBackendBaseURL()).toBe(
      `${window.location.protocol}//api.example.test:3000`,
    );
  });

  it("preserves full origins from env values", () => {
    vi.stubEnv("VITE_BACKEND_BASE_URL", "http://api.example.test:3000/");

    expect(resolveBackendBaseURL()).toBe("http://api.example.test:3000");
  });
});
