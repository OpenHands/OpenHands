import { describe, expect, it } from "vitest";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { isConfigureLlmChecklistItemComplete } from "#/components/features/sidebar/sidebar-onboarding-checklist-llm-complete";

describe("isConfigureLlmChecklistItemComplete", () => {
  it("returns false while LLM readiness is still indeterminate", () => {
    expect(
      isConfigureLlmChecklistItemComplete(
        DEFAULT_SETTINGS,
        false,
        true,
        undefined,
        true,
      ),
    ).toBe(false);
  });

  it("returns true when useLlmConfigured reports configured", () => {
    expect(
      isConfigureLlmChecklistItemComplete(
        DEFAULT_SETTINGS,
        true,
        false,
        undefined,
        false,
      ),
    ).toBe(true);
  });

  it("returns true when any saved LLM profile has an API key", () => {
    expect(
      isConfigureLlmChecklistItemComplete(
        DEFAULT_SETTINGS,
        false,
        true,
        {
          active_profile: "work",
          profiles: [
            {
              name: "work",
              model: "openai/gpt-5.5",
              base_url: "https://api.openai.com/v1",
              api_key_set: true,
            },
          ],
        },
        false,
      ),
    ).toBe(true);
  });

  it("returns true when settings already have a model and API key", () => {
    expect(
      isConfigureLlmChecklistItemComplete(
        {
          ...DEFAULT_SETTINGS,
          llm_api_key_set: true,
          agent_settings: {
            ...(DEFAULT_SETTINGS.agent_settings ?? {}),
            llm: { model: "openai/gpt-5.5" },
          },
        },
        false,
        true,
        undefined,
        true,
      ),
    ).toBe(true);
  });

  it("returns false when no model or auth is present", () => {
    expect(
      isConfigureLlmChecklistItemComplete(
        DEFAULT_SETTINGS,
        false,
        false,
        { active_profile: null, profiles: [] },
        false,
      ),
    ).toBe(false);
  });
});
