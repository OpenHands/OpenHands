/**
 * ProfilesService tests using SDK client mocks.
 *
 * Note: We mock ProfilesClient directly because the SDK's HttpClient uses
 * native fetch that MSW doesn't intercept reliably in Node.js test environments.
 * This approach tests that ProfilesService correctly delegates to ProfilesClient
 * methods and propagates errors from the SDK.
 *
 * For full integration testing, use browser-level tests with MSW.
 */
import { ProfilesClient } from "@openhands/typescript-client/clients";
import { afterEach, describe, expect, it, vi, beforeEach } from "vitest";
import ProfilesService from "#/api/profiles-service/profiles-service.api";

// Mock the ProfilesClient from the SDK using vi.hoisted
const {
  mockListProfiles,
  mockGetProfile,
  mockSaveProfile,
  mockDeleteProfile,
  mockRenameProfile,
  mockActivateProfile,
  mockValidateProfile,
  mockClose,
} = vi.hoisted(() => ({
  mockListProfiles: vi.fn(),
  mockGetProfile: vi.fn(),
  mockSaveProfile: vi.fn(),
  mockDeleteProfile: vi.fn(),
  mockRenameProfile: vi.fn(),
  mockActivateProfile: vi.fn(),
  mockValidateProfile: vi.fn(),
  mockClose: vi.fn(),
}));

vi.mock("@openhands/typescript-client/clients", () => ({
  ProfilesClient: vi.fn(function ProfilesClientMock() {
    return {
      listProfiles: mockListProfiles,
      getProfile: mockGetProfile,
      saveProfile: mockSaveProfile,
      deleteProfile: mockDeleteProfile,
      renameProfile: mockRenameProfile,
      activateProfile: mockActivateProfile,
    };
  }),
  AgentServerClient: vi.fn(function AgentServerClientMock() {
    return { post: mockValidateProfile, close: mockClose };
  }),
}));

vi.mock("#/api/agent-server-client-options", () => ({
  getAgentServerClientOptions: vi.fn(() => ({
    host: "http://localhost:3000",
    apiKey: "test-key",
  })),
}));

describe("ProfilesService", () => {
  beforeEach(() => {
    mockListProfiles.mockReset();
    mockGetProfile.mockReset();
    mockSaveProfile.mockReset();
    mockDeleteProfile.mockReset();
    mockRenameProfile.mockReset();
    mockActivateProfile.mockReset();
    mockValidateProfile.mockReset();
    mockClose.mockReset();
    vi.mocked(ProfilesClient).mockClear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe("listProfiles", () => {
    it("calls listProfiles and returns profiles list", async () => {
      const mockResponse = {
        profiles: [
          {
            name: "gpt-4-profile",
            model: "openai/gpt-4",
            base_url: null,
            api_key_set: true,
          },
          {
            name: "claude-profile",
            model: "anthropic/claude-3",
            base_url: "https://api.anthropic.com",
            api_key_set: false,
          },
        ],
        active_profile: null,
      };

      mockListProfiles.mockResolvedValue(mockResponse);

      const result = await ProfilesService.listProfiles();

      expect(mockListProfiles).toHaveBeenCalled();
      expect(result).toEqual(mockResponse);
      expect(result.profiles).toHaveLength(2);
    });

    it("returns empty profiles array when no profiles exist", async () => {
      const mockResponse = { profiles: [], active_profile: null };
      mockListProfiles.mockResolvedValue(mockResponse);

      const result = await ProfilesService.listProfiles();

      expect(result.profiles).toEqual([]);
    });

    it("propagates errors from the SDK", async () => {
      mockListProfiles.mockRejectedValue(new Error("Network error"));

      await expect(ProfilesService.listProfiles()).rejects.toThrow(
        "Network error",
      );
    });
  });

  describe("getProfile", () => {
    it("calls getProfile with profile name", async () => {
      const mockResponse = {
        name: "my-profile",
        config: { model: "openai/gpt-4" },
        api_key_set: true,
      };

      mockGetProfile.mockResolvedValue(mockResponse);

      const result = await ProfilesService.getProfile("my-profile");

      expect(mockGetProfile).toHaveBeenCalledWith("my-profile", {});
      expect(result).toEqual(mockResponse);
    });

    it("passes exposeSecrets option when set", async () => {
      const mockResponse = {
        name: "my-profile",
        config: { model: "openai/gpt-4", api_key: "encrypted_..." },
        api_key_set: true,
      };

      mockGetProfile.mockResolvedValue(mockResponse);

      await ProfilesService.getProfile("my-profile", "encrypted");

      expect(mockGetProfile).toHaveBeenCalledWith("my-profile", {
        exposeSecrets: "encrypted",
      });
    });
  });

  describe("saveProfile", () => {
    it("calls saveProfile with name and request body", async () => {
      const mockResponse = { name: "new-profile", message: "Profile saved" };
      mockSaveProfile.mockResolvedValue(mockResponse);

      const request = {
        llm: {
          model: "openai/gpt-4",
          api_key: "sk-xxx",
        },
        include_secrets: true,
      };

      const result = await ProfilesService.saveProfile("new-profile", request);

      expect(mockSaveProfile).toHaveBeenCalledWith("new-profile", request);
      expect(result).toEqual(mockResponse);
    });

    it("saves profile with base_url", async () => {
      const mockResponse = { name: "custom-profile", message: "Profile saved" };
      mockSaveProfile.mockResolvedValue(mockResponse);

      const request = {
        llm: {
          model: "openai/gpt-4",
          base_url: "https://custom.api.com",
        },
      };

      await ProfilesService.saveProfile("custom-profile", request);

      expect(mockSaveProfile).toHaveBeenCalledWith("custom-profile", request);
    });
  });

  describe("deleteProfile", () => {
    it("calls deleteProfile with profile name", async () => {
      const mockResponse = { name: "old-profile", message: "Profile deleted" };
      mockDeleteProfile.mockResolvedValue(mockResponse);

      const result = await ProfilesService.deleteProfile("old-profile");

      expect(mockDeleteProfile).toHaveBeenCalledWith("old-profile");
      expect(result).toEqual(mockResponse);
    });
  });

  describe("renameProfile", () => {
    it("calls renameProfile with old and new names", async () => {
      const mockResponse = {
        name: "renamed-profile",
        message: "Profile renamed",
      };
      mockRenameProfile.mockResolvedValue(mockResponse);

      const result = await ProfilesService.renameProfile(
        "old-name",
        "renamed-profile",
      );

      expect(mockRenameProfile).toHaveBeenCalledWith(
        "old-name",
        "renamed-profile",
      );
      expect(result).toEqual(mockResponse);
    });
  });

  describe("activateProfile", () => {
    it("calls activateProfile with profile name", async () => {
      const mockResponse = {
        name: "active-profile",
        message: "Profile activated",
        llm_applied: true,
      };
      mockActivateProfile.mockResolvedValue(mockResponse);

      const result = await ProfilesService.activateProfile("active-profile");

      expect(mockActivateProfile).toHaveBeenCalledWith("active-profile");
      expect(result).toEqual(mockResponse);
    });
  });

  describe("validateProfile", () => {
    const request = { llm: { model: "openai/gpt-4o" }, include_secrets: true };

    it("returns a valid verdict via the typed client", async () => {
      mockValidateProfile.mockResolvedValue({ valid: true });

      const result = await ProfilesService.validateProfile("gpt", request);

      expect(mockValidateProfile).toHaveBeenCalledWith(
        "/api/profiles/gpt/validate",
        request,
        { acceptableStatusCodes: new Set([200]) },
      );
      expect(result).toEqual({ valid: true });
      expect(mockClose).toHaveBeenCalled();
    });

    it("returns null when the agent-server reports 404", async () => {
      mockValidateProfile.mockRejectedValue(
        Object.assign(new Error("HTTP 404"), { status: 404 }),
      );

      await expect(
        ProfilesService.validateProfile("gpt", request),
      ).resolves.toBeNull();
    });

    it("propagates other errors", async () => {
      mockValidateProfile.mockRejectedValue(
        Object.assign(new Error("HTTP 500"), { status: 500 }),
      );

      await expect(
        ProfilesService.validateProfile("gpt", request),
      ).rejects.toThrow("HTTP 500");
    });
  });
});
