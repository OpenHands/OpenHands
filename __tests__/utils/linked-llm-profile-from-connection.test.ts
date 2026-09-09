import { describe, expect, it } from "vitest";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { DEFAULT_OLLAMA_MODEL } from "#/constants/provider-connection-options";
import {
  defaultLlmModelForProvider,
  linkedLlmProfileFromConnection,
} from "#/utils/linked-llm-profile-from-connection";

describe("defaultLlmModelForProvider", () => {
  it("uses the Ollama default for local or remote Ollama", () => {
    expect(defaultLlmModelForProvider("ollama")).toBe(DEFAULT_OLLAMA_MODEL);
  });

  it("falls back to the app default for OpenAI-style keys", () => {
    expect(defaultLlmModelForProvider("openai")).toBe(
      DEFAULT_SETTINGS.llm_model,
    );
  });
});

describe("linkedLlmProfileFromConnection", () => {
  it("names the profile from the connection and links it", () => {
    expect(
      linkedLlmProfileFromConnection({
        id: "conn-ollama",
        display_name: "Office GPU",
        provider: "ollama",
      }),
    ).toEqual({
      name: "Office-GPU",
      request: {
        llm: {
          model: DEFAULT_OLLAMA_MODEL,
          provider_connection_id: "conn-ollama",
        },
      },
    });
  });
});
