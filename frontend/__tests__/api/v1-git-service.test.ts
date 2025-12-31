import { test, expect, vi } from "vitest";
import axios from "axios";
import V1GitService from "../../src/api/git-service/v1-git-service.api";

vi.mock("axios");

test("getGitChanges throws when response is not an array (dead runtime returns HTML)", async () => {
  const htmlResponse = "<!DOCTYPE html><html>...</html>";
  vi.mocked(axios.get).mockResolvedValue({ data: htmlResponse });

  await expect(
    V1GitService.getGitChanges(
      "http://localhost:3000/api/conversations/123",
      "test-api-key",
      "/workspace",
    ),
  ).rejects.toThrow("Invalid response from runtime");
});
