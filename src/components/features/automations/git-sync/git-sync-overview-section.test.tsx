import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { GitSyncStatus } from "#/types/git-sync";
import { GitSyncOverviewSection } from "./git-sync-overview-section";

const baseStatus: GitSyncStatus = {
  enabled: true,
  repo_url: "https://github.com/org/repo.git",
  branch: "main",
  path: "automations",
  encryption_enabled: false,
  interval_seconds: 0,
  last_synced_commit: null,
  last_synced_at: null,
  last_error: null,
  last_error_at: null,
  dirty_count: 0,
};

function renderWith(repoUrl: string) {
  render(
    <GitSyncOverviewSection
      status={{ ...baseStatus, repo_url: repoUrl }}
      onSyncNow={() => {}}
      isSyncing={false}
      syncActivity="idle"
      syncStartedAt={null}
      canManage
    />,
  );
}

describe("GitSyncOverviewSection", () => {
  it("links an https remote, dropping the .git suffix", () => {
    renderWith("https://github.com/org/repo.git");

    const link = screen.getByTestId("git-sync-repo-link");
    expect(link).toHaveAttribute("href", "https://github.com/org/repo");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
    // The configured value stays visible, so what is shown is still what is set.
    expect(link).toHaveTextContent("https://github.com/org/repo.git");
  });

  it("links an ssh remote over https", () => {
    renderWith("git@github.com:org/repo.git");

    expect(screen.getByTestId("git-sync-repo-link")).toHaveAttribute(
      "href",
      "https://github.com/org/repo",
    );
  });

  it("keeps a self-hosted forge's port and forge-specific path", () => {
    renderWith("https://git.example.com:8443/org/team/_git/repo.git");

    expect(screen.getByTestId("git-sync-repo-link")).toHaveAttribute(
      "href",
      "https://git.example.com:8443/org/team/_git/repo",
    );
  });

  // The ssh port says nothing about where the web UI listens, so the rebuilt
  // link drops it -- unlike the https case above, which keeps its own port.
  it("drops the ssh port when rebuilding the browse link over https", () => {
    renderWith("ssh://git@git.example.com:2222/org/repo.git");

    expect(screen.getByTestId("git-sync-repo-link")).toHaveAttribute(
      "href",
      "https://git.example.com/org/repo",
    );
  });

  // A remote configured with credentials in it must not put them in an href,
  // where they would ride along in the outgoing navigation.
  it("strips credentials from the link", () => {
    renderWith("https://user:ghp_token@github.com/org/repo.git");

    const link = screen.getByTestId("git-sync-repo-link");
    expect(link).toHaveAttribute("href", "https://github.com/org/repo");
    expect(link.getAttribute("href")).not.toContain("ghp_token");
  });

  // A bare repo on disk is a valid sync target but has nothing to open.
  it.each(["/tmp/git-sync-remote.git", "not a url", ""])(
    "renders %s as plain text",
    (repoUrl) => {
      renderWith(repoUrl);

      expect(
        screen.queryByTestId("git-sync-repo-link"),
      ).not.toBeInTheDocument();
    },
  );
});
