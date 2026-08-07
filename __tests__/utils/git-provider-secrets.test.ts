import { describe, expect, it } from "vitest";
import {
  allGitProviderSecretNames,
  gitProviderPasswordSecretName,
  gitProviderSshPrivateKeySecretName,
  gitProviderTokenSecretName,
  gitProviderUsernameSecretName,
} from "#/utils/git-provider-secrets";
import { GIT_PROVIDER_ID_PATTERN } from "#/types/git-provider";

describe("git provider secret names", () => {
  it("builds stable secret names for a provider id", () => {
    expect(gitProviderTokenSecretName("github_work")).toBe(
      "GIT_PROVIDER_github_work_TOKEN",
    );
    expect(gitProviderUsernameSecretName("github_work")).toBe(
      "GIT_PROVIDER_github_work_USERNAME",
    );
    expect(gitProviderPasswordSecretName("github_work")).toBe(
      "GIT_PROVIDER_github_work_PASSWORD",
    );
    expect(gitProviderSshPrivateKeySecretName("github_work")).toBe(
      "GIT_PROVIDER_github_work_SSH_PRIVATE_KEY",
    );
    expect(allGitProviderSecretNames("github_work")).toHaveLength(4);
  });

  it("accepts safe provider ids that fit secret name limits", () => {
    expect(GIT_PROVIDER_ID_PATTERN.test("github")).toBe(true);
    expect(GIT_PROVIDER_ID_PATTERN.test("GitLab_1")).toBe(true);
    expect(GIT_PROVIDER_ID_PATTERN.test("1bad")).toBe(false);
    expect(GIT_PROVIDER_ID_PATTERN.test("has-dash")).toBe(false);
  });
});
