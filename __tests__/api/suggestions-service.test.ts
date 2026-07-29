import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import * as cloudSuggestions from "#/api/cloud/suggestions-service.api";
import { SuggestionsService } from "#/api/suggestions-service/suggestions-service.api";
import type { Backend } from "#/api/backend-registry/types";
import type { SuggestedTask } from "#/utils/types";

vi.mock("#/api/cloud/suggestions-service.api", () => ({
  getCloudSuggestedTasks: vi.fn(),
}));

const mockGetCloudSuggestedTasks = vi.mocked(
  cloudSuggestions.getCloudSuggestedTasks,
);

const localBackend: Backend = {
  id: "local",
  name: "Local",
  host: "http://localhost",
  apiKey: "local-key",
  kind: "local",
};

const cloudBackend: Backend = {
  id: "prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer",
  kind: "cloud",
};

beforeEach(() => {
  __resetActiveStoreForTests();
  mockGetCloudSuggestedTasks.mockReset();
});

afterEach(() => {
  __resetActiveStoreForTests();
});

describe("SuggestionsService.getSuggestedTasks", () => {
  it("returns the cloud endpoint items when the active backend is cloud", async () => {
    // Arrange
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });
    const items: SuggestedTask[] = [
      {
        issue_number: 1,
        title: "Fix CI",
        repo: "octo/foo",
        task_type: "FAILING_CHECKS",
        git_provider: "github",
      },
    ];
    mockGetCloudSuggestedTasks.mockResolvedValueOnce({
      items,
      next_page_id: null,
    });

    // Act
    const result = await SuggestionsService.getSuggestedTasks();

    // Assert
    expect(result).toEqual(items);
  });

  it("returns an empty list without calling the cloud endpoint when the active backend is local", async () => {
    // Arrange
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });

    // Act
    const result = await SuggestionsService.getSuggestedTasks();

    // Assert
    expect(result).toEqual([]);
    expect(mockGetCloudSuggestedTasks).not.toHaveBeenCalled();
  });
});
