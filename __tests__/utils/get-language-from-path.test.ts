import { describe, expect, it } from "vitest";
import { getLanguageFromPath } from "#/utils/get-language-from-path";

describe("getLanguageFromPath", () => {
  it.each([
    "Dockerfile",
    "deploy/Dockerfile",
    "/workspace/project/Dockerfile",
    "/workspace/project.v2/DOCKERFILE",
    "C:\\workspace\\project\\Dockerfile",
  ])("recognizes the Dockerfile basename in %s", (path) => {
    expect(getLanguageFromPath(path)).toBe("dockerfile");
  });

  it.each([
    ["src.v2/app.test.TSX", "typescript"],
    ["C:\\project.v2\\src\\app.py", "python"],
    ["deploy/build.dockerfile", "dockerfile"],
    ["deploy/Dockerfile.bak", "text"],
    ["deploy/Dockerfile/config.json", "json"],
    ["/workspace/project.ts/README", "text"],
    ["", "text"],
  ])("uses the final filename to classify %s", (path, language) => {
    expect(getLanguageFromPath(path)).toBe(language);
  });
});
