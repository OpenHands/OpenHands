import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import axios from "axios";
import {
  resolveUiTranslateTarget,
  TranslateService,
} from "#/api/integrations/translate-service";

vi.mock("axios");
vi.mock("#/api/backend-registry/active-store", () => ({
  getEffectiveLocalBackend: () => ({
    id: "default-local",
    host: "http://127.0.0.1:8000",
    apiKey: "session-key",
    kind: "local" as const,
  }),
}));

describe("resolveUiTranslateTarget", () => {
  it("returns pt-BR for Portuguese UI locales", () => {
    expect(resolveUiTranslateTarget("pt")).toBe("pt-BR");
    expect(resolveUiTranslateTarget("pt-BR")).toBe("pt-BR");
  });

  it("returns null for English", () => {
    expect(resolveUiTranslateTarget("en")).toBeNull();
  });
});

describe("TranslateService.translateBatch", () => {
  beforeEach(() => {
    TranslateService.clearCache();
    vi.mocked(axios.post).mockReset();
  });

  afterEach(() => {
    TranslateService.clearCache();
  });

  it("skips the network when the UI language is English", async () => {
    const map = await TranslateService.translateBatch(
      ["Detected use of eval"],
      "en",
    );

    expect(map.get("Detected use of eval")).toBe("Detected use of eval");
    expect(axios.post).not.toHaveBeenCalled();
  });

  it("posts missing texts to the translate proxy and caches the result", async () => {
    vi.mocked(axios.post).mockResolvedValue({
      data: {
        translations: {
          "Detected use of eval": "Uso de eval detectado",
        },
        source: "en",
        target: "pt-BR",
      },
    });

    const first = await TranslateService.translateBatch(
      ["Detected use of eval"],
      "pt",
    );
    expect(first.get("Detected use of eval")).toBe("Uso de eval detectado");
    expect(axios.post).toHaveBeenCalledTimes(1);

    const second = await TranslateService.translateBatch(
      ["Detected use of eval"],
      "pt",
    );
    expect(second.get("Detected use of eval")).toBe("Uso de eval detectado");
    expect(axios.post).toHaveBeenCalledTimes(1);
  });
});
