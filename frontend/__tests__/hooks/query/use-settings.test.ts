import { describe, expect, it, vi } from "vitest";

// Mock external dependencies before importing the module under test
vi.mock("#/context/use-selected-organization", () => ({
  useSelectedOrganizationId: () => ({ organizationId: null }),
}));

vi.mock("#/hooks/use-is-on-intermediate-page", () => ({
  useIsOnIntermediatePage: () => false,
}));

vi.mock("#/hooks/query/use-is-authed", () => ({
  useIsAuthed: () => ({ data: true }),
}));

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => ({ data: { app_mode: "oss" } }),
}));

vi.mock("#/api/organization-service/organization-service.api", () => ({
  organizationService: {
    getOrganizationSettings: vi.fn(),
  },
}));

vi.mock("#/api/settings-service/settings-service.api", () => ({
  default: {
    getSettings: vi.fn(),
  },
}));

// Import the function under test after mocks are set up
import { getSettingsQueryFn } from "#/hooks/query/use-settings";
import { DEFAULT_SETTINGS } from "#/services/settings";
import SettingsService from "#/api/settings-service/settings-service.api";

describe("use-settings: DEFAULT_SETTINGS", () => {
  it("should have stay_logged_in set to true by default", () => {
    expect(DEFAULT_SETTINGS.stay_logged_in).toBe(true);
  });
});

describe("use-settings: normalizeSettingsResponse via getSettingsQueryFn", () => {
  it("should use stay_logged_in from server response when provided as true", async () => {
    vi.mocked(SettingsService.getSettings).mockResolvedValue({
      ...DEFAULT_SETTINGS,
      stay_logged_in: true,
    });

    const result = await getSettingsQueryFn("personal");
    expect(result.stay_logged_in).toBe(true);
  });

  it("should use stay_logged_in from server response when provided as false", async () => {
    vi.mocked(SettingsService.getSettings).mockResolvedValue({
      ...DEFAULT_SETTINGS,
      stay_logged_in: false,
    });

    const result = await getSettingsQueryFn("personal");
    expect(result.stay_logged_in).toBe(false);
  });

  it("should fall back to DEFAULT_SETTINGS.stay_logged_in (true) when not provided in response", async () => {
    const { stay_logged_in: _omitted, ...settingsWithoutStayLoggedIn } =
      DEFAULT_SETTINGS;
    vi.mocked(SettingsService.getSettings).mockResolvedValue(
      settingsWithoutStayLoggedIn as typeof DEFAULT_SETTINGS,
    );

    const result = await getSettingsQueryFn("personal");
    // DEFAULT_SETTINGS.stay_logged_in is true, so fallback should be true
    expect(result.stay_logged_in).toBe(true);
  });

  it("should not override a false stay_logged_in from server with default true", async () => {
    vi.mocked(SettingsService.getSettings).mockResolvedValue({
      ...DEFAULT_SETTINGS,
      stay_logged_in: false,
    });

    const result = await getSettingsQueryFn("personal");
    // Explicitly false from server should not be overridden
    expect(result.stay_logged_in).toBe(false);
  });

  it("should include stay_logged_in in the normalized settings alongside other fields", async () => {
    vi.mocked(SettingsService.getSettings).mockResolvedValue({
      ...DEFAULT_SETTINGS,
      stay_logged_in: false,
      language: "fr",
    });

    const result = await getSettingsQueryFn("personal");
    expect(result.stay_logged_in).toBe(false);
    expect(result.language).toBe("fr");
  });
});