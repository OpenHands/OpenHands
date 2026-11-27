import { describe, expect, it } from "vitest";
import {
  ONBOARDING_PREVIEW_STEP_QUERY_PARAM,
  isOnboardingPreviewActive,
  readOnboardingPreviewStep,
} from "#/components/features/onboarding/onboarding-preview";

describe("onboarding-preview", () => {
  it("reads a valid preview step from the query string", () => {
    expect(
      readOnboardingPreviewStep(`?${ONBOARDING_PREVIEW_STEP_QUERY_PARAM}=3`),
    ).toBe(3);
  });

  it("rejects out-of-range preview steps", () => {
    expect(
      readOnboardingPreviewStep(`?${ONBOARDING_PREVIEW_STEP_QUERY_PARAM}=9`),
    ).toBeNull();
  });

  it("detects when preview mode is active", () => {
    expect(
      isOnboardingPreviewActive(`?${ONBOARDING_PREVIEW_STEP_QUERY_PARAM}=2`),
    ).toBe(true);
    expect(isOnboardingPreviewActive("?foo=bar")).toBe(false);
  });
});
