import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { callCloudProxy } from "#/api/cloud/proxy";
import SettingsService from "#/api/settings-service/settings-service.api";
import { RepositorySelectionForm } from "#/components/features/home/repo-selection-form";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { useHomeStore } from "#/stores/home-store";

vi.mock("#/api/cloud/proxy", () => ({
  callCloudProxy: vi.fn(),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

const cloudBackend: Backend = {
  id: "cloud-ohe",
  name: "OpenHands Enterprise",
  host: "https://ohe.example.com",
  apiKey: "cloud-key",
  kind: "cloud",
};

const WORK_GITLAB = 4;
const SELF_HOSTED_GITLAB = 9;

const connection = (id: number, title: string) => ({
  id,
  title,
  icon: null,
  color: "#fff",
  provider_type: "gitlab",
  connected: true,
  needs_reconnect: false,
});

const repositoryOf = (id: string, fullName: string) => ({
  id,
  full_name: fullName,
  git_provider: "gitlab",
  is_public: false,
  main_branch: "main",
});

const REPOSITORIES_BY_CONNECTION: Record<string, unknown[]> = {
  [WORK_GITLAB]: [repositoryOf("11", "team/work-app")],
  [SELF_HOSTED_GITLAB]: [repositoryOf("22", "team/self-hosted-app")],
  none: [repositoryOf("33", "team/default-app")],
};

function serveCloud({ oauthV2LoginEnabled }: { oauthV2LoginEnabled: boolean }) {
  vi.mocked(callCloudProxy).mockImplementation((async (req: {
    path: string;
  }) => {
    const url = new URL(`https://ohe.example.com${req.path}`);
    switch (url.pathname) {
      case "/api/v1/web-client/config":
        return {
          feature_flags: { enable_oauth_v2_login: oauthV2LoginEnabled },
        };
      case "/oauth/git-providers":
        return [
          connection(WORK_GITLAB, "Work GitLab"),
          connection(SELF_HOSTED_GITLAB, "Self-hosted GitLab"),
        ];
      case "/api/v1/git/repositories/search":
        return {
          items:
            REPOSITORIES_BY_CONNECTION[
              url.searchParams.get("git_provider_id") ?? "none"
            ],
          next_page_id: null,
        };
      case "/api/v1/git/branches/search":
        return {
          items: [{ name: "main", commit_sha: "abc", protected: false }],
          next_page_id: null,
        };
      case "/api/v1/git/installations/search":
        return { items: [], next_page_id: null };
      default:
        throw new Error(`Unexpected cloud request ${req.path}`);
    }
  }) as never);
}

const requestedUrls = (prefix: string) =>
  vi
    .mocked(callCloudProxy)
    .mock.calls.map(
      (call) => new URL(`https://x${(call[0] as { path: string }).path}`),
    )
    .filter((url) => url.pathname === prefix);

const renderForm = (onConfirm = vi.fn()) => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(
    <QueryClientProvider client={client}>
      <ActiveBackendProvider>
        <RepositorySelectionForm onConfirm={onConfirm} />
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
  return onConfirm;
};

describe("RepositorySelectionForm with two GitLab connections", () => {
  beforeEach(() => {
    window.localStorage.clear();
    __resetActiveStoreForTests();
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: null });
    useHomeStore.setState({
      recentRepositories: [],
      lastSelectedProvider: null,
      lastSelectedGitProviderId: null,
      repositorySelections: {},
    });
    vi.mocked(callCloudProxy).mockReset();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...DEFAULT_SETTINGS,
      provider_tokens_set: { gitlab: null },
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    __resetActiveStoreForTests();
  });

  it("lists, branches and confirms through the connection that was picked", async () => {
    serveCloud({ oauthV2LoginEnabled: true });
    const user = userEvent.setup();
    const onConfirm = renderForm();

    await user.click(await screen.findByTestId("git-provider-dropdown"));
    await user.click(
      await screen.findByRole("option", { name: "Self-hosted GitLab" }),
    );

    await user.click(screen.getByTestId("git-repo-dropdown"));
    await user.click(await screen.findByText("team/self-hosted-app"));

    await waitFor(() =>
      expect(screen.getByTestId("repo-launch-button")).toBeEnabled(),
    );
    await user.click(screen.getByTestId("repo-launch-button"));

    expect(onConfirm).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({
        provider: "gitlab",
        repository: expect.objectContaining({
          full_name: "team/self-hosted-app",
          git_provider_id: SELF_HOSTED_GITLAB,
        }),
        branch: expect.objectContaining({ name: "main" }),
      }),
    );
    expect(
      requestedUrls("/api/v1/git/repositories/search").map((url) =>
        url.searchParams.get("git_provider_id"),
      ),
    ).not.toContain(String(WORK_GITLAB));
    expect(
      requestedUrls("/api/v1/git/branches/search").every(
        (url) =>
          url.searchParams.get("git_provider_id") ===
          String(SELF_HOSTED_GITLAB),
      ),
    ).toBe(true);
    expect(useHomeStore.getState().lastSelectedGitProviderId).toBe(
      SELF_HOSTED_GITLAB,
    );
  });

  it("switches to the other connection's repositories when it is picked", async () => {
    serveCloud({ oauthV2LoginEnabled: true });
    const user = userEvent.setup();
    renderForm();

    await user.click(await screen.findByTestId("git-provider-dropdown"));
    await user.click(
      await screen.findByRole("option", { name: "Work GitLab" }),
    );
    await user.click(screen.getByTestId("git-repo-dropdown"));
    expect(await screen.findByText("team/work-app")).toBeInTheDocument();
    expect(screen.queryByText("team/self-hosted-app")).not.toBeInTheDocument();

    await user.click(screen.getByTestId("git-provider-dropdown"));
    await user.click(
      await screen.findByRole("option", { name: "Self-hosted GitLab" }),
    );
    await user.click(screen.getByTestId("git-repo-dropdown"));

    expect(await screen.findByText("team/self-hosted-app")).toBeInTheDocument();
    expect(screen.queryByText("team/work-app")).not.toBeInTheDocument();
  });

  it("sends no git_provider_id and offers no connection picker without the flag", async () => {
    serveCloud({ oauthV2LoginEnabled: false });
    const user = userEvent.setup();
    const onConfirm = renderForm();

    await user.click(await screen.findByTestId("git-repo-dropdown"));
    await user.click(await screen.findByText("team/default-app"));
    await waitFor(() =>
      expect(screen.getByTestId("repo-launch-button")).toBeEnabled(),
    );
    await user.click(screen.getByTestId("repo-launch-button"));

    expect(
      screen.queryByTestId("git-provider-dropdown"),
    ).not.toBeInTheDocument();
    expect(onConfirm).toHaveBeenCalledOnce();
    expect(onConfirm.mock.calls[0][0].repository).not.toHaveProperty(
      "git_provider_id",
    );
    const gitUrls = vi
      .mocked(callCloudProxy)
      .mock.calls.map((call) => (call[0] as { path: string }).path)
      .filter((path) => path.startsWith("/api/v1/git/"));
    expect(gitUrls.length).toBeGreaterThan(0);
    expect(gitUrls.some((path) => path.includes("git_provider_id"))).toBe(
      false,
    );
  });
});
