import { describe, it, expect, vi, beforeEach } from "vitest";
import { openHands } from "#/api/open-hands-axios";
import ProfilesService from "#/api/settings-service/profiles-service.api";

vi.mock("#/api/open-hands-axios", () => ({
  openHands: {
    get: vi.fn(),
    post: vi.fn(),
    delete: vi.fn(),
  },
}));

describe("ProfilesService", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("listProfiles", () => {
    it("GETs /api/v1/settings/profiles and returns the body", async () => {
      const body = {
        profiles: [
          {
            name: "openai_gpt-4o",
            model: "openai/gpt-4o",
            base_url: null,
            api_key_set: true,
          },
        ],
        active_profile: "openai_gpt-4o",
      };
      vi.mocked(openHands.get).mockResolvedValue({ data: body });

      await expect(ProfilesService.listProfiles()).resolves.toEqual(body);
      expect(openHands.get).toHaveBeenCalledWith("/api/v1/settings/profiles");
    });
  });

  describe("saveProfile", () => {
    it("POSTs the name-encoded URL with the provided request body", async () => {
      vi.mocked(openHands.post).mockResolvedValue({ data: {} });

      await ProfilesService.saveProfile("openai_gpt-4o", {
        include_secrets: true,
        llm: { model: "openai/gpt-4o" },
      });

      expect(openHands.post).toHaveBeenCalledWith(
        "/api/v1/settings/profiles/openai_gpt-4o",
        { include_secrets: true, llm: { model: "openai/gpt-4o" } },
      );
    });

    it("defaults to an empty body when none is provided", async () => {
      vi.mocked(openHands.post).mockResolvedValue({ data: {} });

      await ProfilesService.saveProfile("my-profile");

      expect(openHands.post).toHaveBeenCalledWith(
        "/api/v1/settings/profiles/my-profile",
        {},
      );
    });

    it("URL-encodes the name segment", async () => {
      vi.mocked(openHands.post).mockResolvedValue({ data: {} });

      // Even though the backend rejects "/" in names, any future-legal
      // unicode should survive round-tripping through encodeURIComponent.
      await ProfilesService.saveProfile("my profile");

      expect(openHands.post).toHaveBeenCalledWith(
        "/api/v1/settings/profiles/my%20profile",
        {},
      );
    });
  });

  describe("deleteProfile", () => {
    it("DELETEs the name-encoded URL", async () => {
      vi.mocked(openHands.delete).mockResolvedValue({ data: {} });

      await ProfilesService.deleteProfile("openai_gpt-4o");

      expect(openHands.delete).toHaveBeenCalledWith(
        "/api/v1/settings/profiles/openai_gpt-4o",
      );
    });
  });

  describe("activateProfile", () => {
    it("POSTs to the /activate sub-resource", async () => {
      vi.mocked(openHands.post).mockResolvedValue({ data: {} });

      await ProfilesService.activateProfile("openai_gpt-4o");

      expect(openHands.post).toHaveBeenCalledWith(
        "/api/v1/settings/profiles/openai_gpt-4o/activate",
      );
    });
  });

  describe("renameProfile", () => {
    it("POSTs to /rename with the new_name body", async () => {
      vi.mocked(openHands.post).mockResolvedValue({ data: {} });

      await ProfilesService.renameProfile("old", "new");

      expect(openHands.post).toHaveBeenCalledWith(
        "/api/v1/settings/profiles/old/rename",
        { new_name: "new" },
      );
    });
  });

  describe("error propagation", () => {
    // The hooks that call these methods rely on promise rejection to route
    // errors into mutateWithToast — if a method ever swallows an axios
    // error, toasts stop working and callers lose visibility.
    it("rejects when listProfiles fails", async () => {
      const err = new Error("network down");
      vi.mocked(openHands.get).mockRejectedValue(err);
      await expect(ProfilesService.listProfiles()).rejects.toThrow(
        "network down",
      );
    });

    it("rejects when saveProfile fails", async () => {
      vi.mocked(openHands.post).mockRejectedValue(new Error("409 conflict"));
      await expect(ProfilesService.saveProfile("x")).rejects.toThrow(
        "409 conflict",
      );
    });

    it("rejects when deleteProfile fails", async () => {
      vi.mocked(openHands.delete).mockRejectedValue(new Error("boom"));
      await expect(ProfilesService.deleteProfile("x")).rejects.toThrow("boom");
    });

    it("rejects when activateProfile fails", async () => {
      vi.mocked(openHands.post).mockRejectedValue(new Error("not found"));
      await expect(ProfilesService.activateProfile("x")).rejects.toThrow(
        "not found",
      );
    });

    it("rejects when renameProfile fails", async () => {
      vi.mocked(openHands.post).mockRejectedValue(new Error("already exists"));
      await expect(ProfilesService.renameProfile("a", "b")).rejects.toThrow(
        "already exists",
      );
    });
  });
});
