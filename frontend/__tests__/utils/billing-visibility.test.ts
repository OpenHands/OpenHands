import { describe, it, expect } from "vitest";
import { isBillingHidden } from "#/utils/org/billing-visibility";
import { WebClientConfig } from "#/api/option-service/option.types";

describe("isBillingHidden", () => {
  const createConfig = (
    featureFlagOverrides: Partial<WebClientConfig["feature_flags"]> = {},
  ): WebClientConfig =>
    ({
      app_mode: "saas",
      posthog_client_key: "test",
      feature_flags: {
        enable_billing: true,
        hide_llm_settings: false,
        enable_jira: false,
        enable_jira_dc: false,
        enable_linear: false,
        ...featureFlagOverrides,
      },
    }) as WebClientConfig;

  it("should return true when config is undefined (safe default)", () => {
    expect(isBillingHidden(undefined, true)).toBe(true);
  });

  it("should return true when enable_billing is false", () => {
    const config = createConfig({ enable_billing: false });
    expect(isBillingHidden(config, true)).toBe(true);
  });

  it("should return true when user lacks view_billing permission", () => {
    const config = createConfig();
    expect(isBillingHidden(config, false)).toBe(true);
  });

  it("should return true when both enable_billing is false and user lacks permission", () => {
    const config = createConfig({ enable_billing: false });
    expect(isBillingHidden(config, false)).toBe(true);
  });

  it("should return false when enable_billing is true and user has view_billing permission", () => {
    const config = createConfig();
    expect(isBillingHidden(config, true)).toBe(false);
  });

  it("should treat enable_billing as true by default (billing visible, subject to permission)", () => {
    const config = createConfig({ enable_billing: true });
    expect(isBillingHidden(config, true)).toBe(false);
  });
});
