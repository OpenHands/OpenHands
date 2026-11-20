import { describe, expect, it } from "vitest";
import fs from "fs";
import path from "path";
import { ONBOARDING_DEFAULT_LLM_MODEL } from "#/components/features/onboarding/steps/setup-llm-step";

// Display names for models the ONBOARDING$LLM_SUBTITLE string may advertise.
// When the onboarding default changes, add the new model here and rewrite the
// subtitle in every locale. A missing entry is the tripwire: it means the
// default moved without anyone deciding what the subtitle should say.
const MODEL_DISPLAY_NAMES: Record<string, string> = {
  "openai/gpt-5.6-sol": "OpenAI GPT-5.6 Sol",
};

// Regression coverage for #16914. The onboarding default has moved twice
// (#16657: GLM 5.2 -> Kimi K3, #16922: Kimi K3 -> GPT-5.6 Sol), and the first
// switch shipped without touching ONBOARDING$LLM_SUBTITLE, so the step said
// GLM-5.2 was pre-selected while the model field showed kimi-k3. Both
// switches synced the subtitle by hand; this test makes the sync mandatory.
// It checks the source-of-truth (translation.json) rather than the rendered
// subtitle, because the test environment's i18next mock returns keys instead
// of translated strings.
describe("ONBOARDING$LLM_SUBTITLE", () => {
  const translationPath = path.join(
    __dirname,
    "../../src/i18n/translation.json",
  );
  const translation = JSON.parse(
    fs.readFileSync(translationPath, "utf-8"),
  ) as Record<string, Record<string, string>>;

  it("names the pre-selected default model in every locale", () => {
    const displayName = MODEL_DISPLAY_NAMES[ONBOARDING_DEFAULT_LLM_MODEL];
    expect(
      displayName,
      `ONBOARDING_DEFAULT_LLM_MODEL is "${ONBOARDING_DEFAULT_LLM_MODEL}", ` +
        "which has no display name registered in this test. Add it to " +
        "MODEL_DISPLAY_NAMES and update ONBOARDING$LLM_SUBTITLE in every " +
        "locale of src/i18n/translation.json to name the new default.",
    ).toBeDefined();

    const subtitles = translation.ONBOARDING$LLM_SUBTITLE;
    expect(subtitles).toBeDefined();
    Object.entries(subtitles).forEach(([locale, subtitle]) => {
      expect(
        subtitle,
        `locale "${locale}" should mention ${displayName}`,
      ).toContain(displayName);
    });
  });
});
