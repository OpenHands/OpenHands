import { describe, expect, it } from "vitest";
import {
  getPathBasename,
  looksLikeWorkspaceFilePath,
  stripWorkspacePrefix,
  toFilesTabPath,
} from "#/utils/path-utils";
import { DEFAULT_WORKING_DIR } from "#/api/agent-server-config";
import { getGitPath } from "#/utils/get-git-path";
import { getLanguageFromPath } from "#/utils/get-language-from-path";
import { isMarkdownFilePath } from "#/utils/is-markdown-file-path";

describe("path-utils", () => {
  describe("getPathBasename", () => {
    it("returns an empty string for empty or whitespace-only input", () => {
      expect(getPathBasename("")).toBe("");
      expect(getPathBasename("   ")).toBe("");
    });

    it("handles POSIX paths with and without trailing slashes", () => {
      expect(getPathBasename("/workspace/project/agent-canvas")).toBe(
        "agent-canvas",
      );
      expect(getPathBasename("/workspace/project/agent-canvas/")).toBe(
        "agent-canvas",
      );
    });

    it("handles Windows-style paths", () => {
      expect(getPathBasename("C:\\Users\\me\\repo")).toBe("repo");
      expect(getPathBasename("C:\\Users\\me\\repo\\")).toBe("repo");
    });

    it("returns an empty string for root paths", () => {
      expect(getPathBasename("/")).toBe("");
      expect(getPathBasename("C:\\")).toBe("");
    });

    it("preserves relative basenames", () => {
      expect(getPathBasename("repo")).toBe("repo");
      expect(getPathBasename("./repo")).toBe("repo");
    });
  });

  describe("stripWorkspacePrefix", () => {
    it("removes the /workspace/<name>/ prefix when present", () => {
      expect(stripWorkspacePrefix("/workspace/repo/src/file.py")).toBe(
        "src/file.py",
      );
      expect(
        stripWorkspacePrefix("/workspace/my-project/components/Button.tsx"),
      ).toBe("components/Button.tsx");
    });

    it("returns an empty string when only the workspace root has a trailing slash", () => {
      expect(stripWorkspacePrefix("/workspace/repo/")).toBe("");
    });

    it("leaves non-workspace or incomplete paths unchanged", () => {
      expect(stripWorkspacePrefix("/workspace")).toBe("/workspace");
      expect(stripWorkspacePrefix("/workspace/repo")).toBe("/workspace/repo");
      expect(stripWorkspacePrefix("relative/path.ts")).toBe("relative/path.ts");
      expect(stripWorkspacePrefix("")).toBe("");
    });
  });

  describe("toFilesTabPath", () => {
    it("strips workspace and working-dir prefixes", () => {
      expect(toFilesTabPath("/workspace/repo/src/a.ts")).toBe("src/a.ts");
      expect(toFilesTabPath("/Users/me/ws/a.ts", "/Users/me/ws")).toBe("a.ts");
    });

    it("aliases toWorkspaceRelativePath for Files-drawer callers", () => {
      expect(toFilesTabPath("/workspace/project/a.ts")).toBe("a.ts");
      expect(toFilesTabPath("src/a.ts:12-40", "/workspace/project")).toBe(
        "src/a.ts",
      );
    });
  });

  describe("looksLikeWorkspaceFilePath", () => {
    it("accepts common workspace file tokens", () => {
      expect(looksLikeWorkspaceFilePath("test.md")).toBe(true);
      expect(looksLikeWorkspaceFilePath("src/app.ts")).toBe(true);
      expect(looksLikeWorkspaceFilePath("docs/guide.md:12")).toBe(true);
    });

    it("rejects non-path dotted tokens, versions, MIME types, and URLs", () => {
      expect(looksLikeWorkspaceFilePath("console.log")).toBe(false);
      expect(looksLikeWorkspaceFilePath("v1.2.3")).toBe(false);
      expect(looksLikeWorkspaceFilePath("application/json")).toBe(false);
      expect(looksLikeWorkspaceFilePath("https://example.com/a.ts")).toBe(
        false,
      );
      expect(looksLikeWorkspaceFilePath("bareword")).toBe(false);
    });
  });
});

describe("get-git-path", () => {
  describe("getGitPath", () => {
    it("should return the default working dir when no repository is selected", () => {
      expect(getGitPath(null)).toBe(DEFAULT_WORKING_DIR);
      expect(getGitPath(undefined)).toBe(DEFAULT_WORKING_DIR);
    });

    it("should handle standard owner/repo format (GitHub)", () => {
      expect(getGitPath("OpenHands/OpenHands")).toBe(
        `${DEFAULT_WORKING_DIR}/OpenHands`,
      );
      expect(getGitPath("facebook/react")).toBe(`${DEFAULT_WORKING_DIR}/react`);
    });

    it("should handle nested group paths (GitLab)", () => {
      expect(getGitPath("modernhealth/frontend-guild/pan")).toBe(
        `${DEFAULT_WORKING_DIR}/pan`,
      );
      expect(getGitPath("group/subgroup/repo")).toBe(
        `${DEFAULT_WORKING_DIR}/repo`,
      );
      expect(getGitPath("a/b/c/d/repo")).toBe(`${DEFAULT_WORKING_DIR}/repo`);
    });

    it("should handle single segment paths", () => {
      expect(getGitPath("repo")).toBe(`${DEFAULT_WORKING_DIR}/repo`);
    });

    it("should handle empty string", () => {
      expect(getGitPath("")).toBe(DEFAULT_WORKING_DIR);
    });

    describe("with a backend-provided workspace path", () => {
      it("prefers the explicit workspace path over derived git paths", () => {
        expect(
          getGitPath(
            "OpenHands/software-agent-sdk",
            "/workspace/project/agent-canvas",
          ),
        ).toBe("/workspace/project/agent-canvas");
      });

      it("ignores blank workspace paths and falls back to heuristics", () => {
        expect(getGitPath("OpenHands/software-agent-sdk", "  ")).toBe(
          `${DEFAULT_WORKING_DIR}/software-agent-sdk`,
        );
      });
    });
  });
});

describe("get-language-from-path", () => {
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
});

describe("is-markdown-file-path", () => {
  describe("isMarkdownFilePath", () => {
    it("matches markdown extensions", () => {
      expect(isMarkdownFilePath("canvas.md")).toBe(true);
      expect(isMarkdownFilePath("/workspace/project/report.markdown")).toBe(
        true,
      );
      expect(isMarkdownFilePath("docs/page.mdx")).toBe(true);
    });

    it("matches extensions case-insensitively", () => {
      expect(isMarkdownFilePath("README.MD")).toBe(true);
      expect(isMarkdownFilePath("notes.Markdown")).toBe(true);
    });

    it("uses only the last extension of dotted names", () => {
      expect(isMarkdownFilePath("foo.bar.md")).toBe(true);
      expect(isMarkdownFilePath("report.md.bak")).toBe(false);
    });

    it("rejects paths without a markdown extension", () => {
      expect(isMarkdownFilePath("Makefile")).toBe(false);
      expect(isMarkdownFilePath("/workspace/app.ts")).toBe(false);
      expect(isMarkdownFilePath("")).toBe(false);
    });
  });
});
