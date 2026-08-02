import { describe, expect, it } from "vitest";
import { http, HttpResponse } from "msw";
import ConfigService from "#/api/config-service/config-service.api";
import { server } from "#/mocks/node";

describe("ConfigService", () => {
  it("derives providers from llm endpoints", async () => {
    const page = await ConfigService.searchProviders({ limit: 10 });

    expect(page.next_page_id).toBeNull();
    expect(page.items.some((provider) => provider.name === "anthropic")).toBe(
      true,
    );
    expect(
      page.items.find((provider) => provider.name === "anthropic")?.verified,
    ).toBe(true);
  });

  it("derives provider models from llm endpoints", async () => {
    const page = await ConfigService.searchModels({
      provider__eq: "anthropic",
      limit: 20,
    });

    expect(page.next_page_id).toBeNull();
    expect(
      page.items.some((model) => model.name === "claude-opus-4-5-20251101"),
    ).toBe(true);
    expect(page.items.every((model) => model.provider === "anthropic")).toBe(
      true,
    );
  });

  it("includes verified providers absent from /api/llm/providers and keeps them within the limit", async () => {
    // Arrange: mirror the real local agent-server, where
    // /api/llm/providers comes from litellm (no "openhands"),
    // but /api/llm/models/verified has "openhands" as a key.
    const litellmOnlyProviders = Array.from(
      { length: 10 },
      (_, i) => `litellm_provider_${i}`,
    );
    server.use(
      http.get("/api/llm/providers", () =>
        HttpResponse.json({ providers: litellmOnlyProviders }),
      ),
      http.get("/api/llm/models/verified", () =>
        HttpResponse.json({
          models: {
            openhands: ["claude-opus-4-7", "gpt-5.5"],
            anthropic: ["claude-opus-4-5-20251101"],
          },
        }),
      ),
    );

    // Act: request fewer items than the litellm provider count to also
    // exercise the ordering fix (verified providers must come first so
    // they survive limitItems).
    const page = await ConfigService.searchProviders({ limit: 3 });

    // Assert
    const openhands = page.items.find((p) => p.name === "openhands");
    expect(openhands).toEqual({ name: "openhands", verified: true });
  });

  it("includes providers only discoverable from model IDs (e.g. openrouter)", async () => {
    // Arrange: mirror the real local agent-server, where "openrouter" is
    // absent from both /api/llm/providers and /api/llm/models/verified but
    // its models appear in /api/llm/models as "openrouter/<model>" IDs.
    server.use(
      http.get("/api/llm/providers", () =>
        HttpResponse.json({ providers: ["anthropic"] }),
      ),
      http.get("/api/llm/models", () =>
        HttpResponse.json({
          models: [
            "anthropic/claude-opus-4-5-20251101",
            "openrouter/anthropic/claude-opus-4-5",
            "openrouter/openai/gpt-5.5",
            "standalone-model",
          ],
        }),
      ),
      http.get("/api/llm/models/verified", () =>
        HttpResponse.json({
          models: { anthropic: ["claude-opus-4-5-20251101"] },
        }),
      ),
    );

    // Act
    const page = await ConfigService.searchProviders({ limit: 10 });

    // Assert: openrouter is surfaced (unverified), providers stay
    // deduplicated, and bare model IDs don't become providers.
    const openrouter = page.items.find((p) => p.name === "openrouter");
    expect(openrouter).toEqual({ name: "openrouter", verified: false });
    expect(page.items.filter((p) => p.name === "anthropic")).toHaveLength(1);
    expect(page.items.some((p) => p.name === "standalone-model")).toBe(false);
  });
});
