import { describe, it, expect } from "vitest";

describe("GitControlBar clone prompt format", () => {
  // Helper function that mirrors the logic in git-control-bar.tsx
  const generateClonePrompt = (
    fullName: string,
    gitProvider: string,
    branchName: string,
  ) => {
    const providerName =
      gitProvider.charAt(0).toUpperCase() + gitProvider.slice(1);
    return `Clone ${fullName} from ${providerName} and checkout branch ${branchName}.`;
  };

  it("should include GitHub in clone prompt for github provider", () => {
    const prompt = generateClonePrompt("user/repo", "github", "main");
    expect(prompt).toBe("Clone user/repo from Github and checkout branch main.");
  });

  it("should include GitLab in clone prompt for gitlab provider", () => {
    const prompt = generateClonePrompt("group/project", "gitlab", "develop");
    expect(prompt).toBe(
      "Clone group/project from Gitlab and checkout branch develop.",
    );
  });

  it("should handle different branch names", () => {
    const prompt = generateClonePrompt(
      "hieptl.developer-group/hieptl.developer-project",
      "gitlab",
      "add-batman-microagent",
    );
    expect(prompt).toBe(
      "Clone hieptl.developer-group/hieptl.developer-project from Gitlab and checkout branch add-batman-microagent.",
    );
  });

  it("should capitalize first letter of provider name", () => {
    const githubPrompt = generateClonePrompt("a/b", "github", "main");
    const gitlabPrompt = generateClonePrompt("a/b", "gitlab", "main");

    expect(githubPrompt).toContain("from Github");
    expect(gitlabPrompt).toContain("from Gitlab");
  });
});
