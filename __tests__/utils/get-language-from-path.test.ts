import { describe, expect, it } from "vitest";
import { getLanguageFromPath } from "#/utils/get-language-from-path";

describe("getLanguageFromPath", () => {
  it.each([
    ["index.js", "javascript"],
    ["component.JSX", "javascript"],
    ["types.ts", "typescript"],
    ["view.tsx", "typescript"],
    ["script.py", "python"],
    ["index.html", "html"],
    ["styles.css", "css"],
    ["package.json", "json"],
    ["README.md", "markdown"],
    ["workflow.yml", "yaml"],
    ["config.yaml", "yaml"],
    ["install.sh", "bash"],
    ["profile.bash", "bash"],
    ["container.dockerfile", "dockerfile"],
    ["main.rs", "rust"],
    ["main.go", "go"],
    ["Main.java", "java"],
    ["main.cpp", "cpp"],
    ["legacy.cc", "cpp"],
    ["native.cxx", "cpp"],
    ["native.c", "c"],
    ["task.rb", "ruby"],
    ["index.php", "php"],
    ["query.sql", "sql"],
    ["README", "text"],
    ["archive.unknown", "text"],
    ["", "text"],
  ])("maps %s to %s", (path, language) => {
    expect(getLanguageFromPath(path)).toBe(language);
  });

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
  ])("uses the final filename to classify %s", (path, language) => {
    expect(getLanguageFromPath(path)).toBe(language);
  });
});
