// @vitest-environment node
import { describe, expect, it, vi } from "vitest";

import {
  MarsApiError,
  buildOpenHandsManifest,
  createMarsApiClient,
} from "../../scripts/mars-api.mjs";

const API_URL = "https://api.example.test";

function clientAnswering(body: unknown) {
  const fetchImpl = vi.fn(
    async () =>
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
  );
  const client = createMarsApiClient({
    getToken: async () => "token",
    baseUrl: API_URL,
    fetchImpl: fetchImpl as unknown as typeof fetch,
  });
  return { client, fetchImpl };
}

describe("listSandboxSizes", () => {
  it("reads the size catalog from the sessions sandbox endpoint", async () => {
    const sizes = [
      {
        slug: "mars-2vcpu-4gb",
        vcpus: 2,
        memory_mb: 4096,
        price_per_hour_usd: 0.036,
        price_per_month_usd: 24,
      },
    ];
    const { client, fetchImpl } = clientAnswering({ sizes });

    const result = await client.listSandboxSizes();

    expect(result).toEqual(sizes);
    expect(fetchImpl).toHaveBeenCalledWith(
      `${API_URL}/v2/agents/sessions/sandbox/sizes`,
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("answers an empty list when the body has no sizes", async () => {
    const { client } = clientAnswering({});

    expect(await client.listSandboxSizes()).toEqual([]);
  });
});

describe("buildOpenHandsManifest", () => {
  it("writes the sandbox size as a top-level manifest key", () => {
    expect(buildOpenHandsManifest({ size: "mars-16vcpu-32gb" })).toBe(
      [
        "agent: openhands",
        "template: openhands",
        "keep_warm: false",
        "size: mars-16vcpu-32gb",
        "",
      ].join("\n"),
    );
  });

  it("leaves the size to DigitalOcean when none is given", () => {
    expect(buildOpenHandsManifest({ size: "" })).not.toMatch(/size:/);
    expect(buildOpenHandsManifest()).not.toMatch(/size:/);
  });

  it.each([
    "mars-2vcpu-4gb\nsecrets:",
    "Mars-2vcpu",
    "-leading-dash",
    "has space",
    "a".repeat(64),
  ])("rejects %j as a size so nothing can be injected", (size) => {
    expect(() => buildOpenHandsManifest({ size })).toThrow(MarsApiError);
  });
});
