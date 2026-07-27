import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Backend } from "#/api/backend-registry/types";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { SecretsService } from "#/api/secrets-service";

const {
  mockGetSecret,
  mockUpsertSecret,
  mockDeleteSecret,
} = vi.hoisted(() => ({
  mockGetSecret: vi.fn(),
  mockUpsertSecret: vi.fn(),
  mockDeleteSecret: vi.fn(),
}));

vi.mock("@openhands/typescript-client/clients", () => ({
  SettingsClient: vi.fn(function SettingsClientMock() {
    return {
      getSecret: mockGetSecret,
      upsertSecret: mockUpsertSecret,
      deleteSecret: mockDeleteSecret,
    };
  }),
}));

const localBackend: Backend = {
  id: "local",
  name: "Local",
  host: "http://127.0.0.1:8000",
  apiKey: "",
  kind: "local",
};

describe("SecretsService", () => {
  beforeEach(() => {
    window.localStorage.clear();
    __resetActiveStoreForTests();
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    mockGetSecret.mockReset();
    mockUpsertSecret.mockReset();
    mockDeleteSecret.mockReset();
  });

  afterEach(() => {
    __resetActiveStoreForTests();
  });

  it("preserves the existing value when updating description", async () => {
    // Arrange
    mockGetSecret.mockResolvedValue("keep-this");
    mockUpsertSecret.mockResolvedValue({
      name: "OpenAI_API_Key",
      description: "Updated description",
    });

    // Act
    await SecretsService.updateSecret(
      "OpenAI_API_Key",
      "OpenAI_API_Key",
      "Updated description",
    );

    // Assert
    expect(mockGetSecret).toHaveBeenCalledWith("OpenAI_API_Key");
    expect(mockUpsertSecret).toHaveBeenCalledWith({
      name: "OpenAI_API_Key",
      value: "keep-this",
      description: "Updated description",
    });
    expect(mockDeleteSecret).not.toHaveBeenCalled();
  });

  it("renames secrets by re-upserting and removing the old entry", async () => {
    // Arrange
    mockGetSecret.mockResolvedValue("original-value");
    mockUpsertSecret.mockResolvedValue({
      name: "New_Key",
      description: "Renamed",
    });
    mockDeleteSecret.mockResolvedValue({ deleted: true });

    // Act
    await SecretsService.updateSecret("Old_Key", "New_Key", "Renamed");

    // Assert
    expect(mockGetSecret).toHaveBeenCalledWith("Old_Key");
    expect(mockUpsertSecret).toHaveBeenCalledWith({
      name: "New_Key",
      value: "original-value",
      description: "Renamed",
    });
    expect(mockDeleteSecret).toHaveBeenCalledWith("Old_Key");
  });
});
