import { render, screen, fireEvent } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { I18nKey } from "#/i18n/declaration";
import type { GitSyncStatus } from "#/types/git-sync";
import { GitSyncConfigForm } from "./git-sync-config-form";

const mutate = vi.fn();

vi.mock("#/hooks/query/use-git-sync", () => ({
  useUpdateGitSyncConfig: () => ({ mutate, isPending: false }),
}));

afterEach(() => {
  vi.clearAllMocks();
});

const baseStatus: GitSyncStatus = {
  enabled: true,
  repo_url: "https://example.com/org/repo.git",
  branch: "main",
  path: "automations",
  encryption_enabled: false,
  interval_seconds: 0,
  last_synced_commit: "abc1234",
  last_synced_at: "2026-08-10T00:00:00Z",
  last_error: null,
  last_error_at: null,
  dirty_count: 0,
};

describe("GitSyncConfigForm", () => {
  it("always renders the token and encryption key fields empty", () => {
    render(<GitSyncConfigForm status={baseStatus} canManage />);

    expect(screen.getByTestId("git-sync-token-input")).toHaveValue("");
    expect(screen.getByTestId("git-sync-encryption-key-input")).toHaveValue("");
  });

  it("shows the encryption key placeholder based on encryption_enabled", () => {
    const { rerender } = render(
      <GitSyncConfigForm status={baseStatus} canManage />,
    );
    expect(screen.getByTestId("git-sync-encryption-key-input")).toHaveAttribute(
      "placeholder",
      I18nKey.AUTOMATIONS$GIT_SYNC$KEY_UNSET_PLACEHOLDER,
    );

    rerender(
      <GitSyncConfigForm
        status={{ ...baseStatus, encryption_enabled: true }}
        canManage
      />,
    );
    expect(screen.getByTestId("git-sync-encryption-key-input")).toHaveAttribute(
      "placeholder",
      I18nKey.AUTOMATIONS$GIT_SYNC$KEY_SET_PLACEHOLDER,
    );
  });

  it("enables submit once a field changes and sends only the changed field", () => {
    render(<GitSyncConfigForm status={baseStatus} canManage />);

    expect(screen.getByTestId("git-sync-save-button")).toBeDisabled();

    fireEvent.change(screen.getByTestId("git-sync-branch-input"), {
      target: { value: "develop" },
    });
    expect(screen.getByTestId("git-sync-save-button")).not.toBeDisabled();

    fireEvent.click(screen.getByTestId("git-sync-save-button"));

    expect(mutate).toHaveBeenCalledTimes(1);
    expect(mutate.mock.calls[0][0]).toEqual({ branch: "develop" });
  });

  it("sends a typed token as a plain string, omitting unrelated fields", () => {
    render(<GitSyncConfigForm status={baseStatus} canManage />);

    fireEvent.change(screen.getByTestId("git-sync-token-input"), {
      target: { value: "ghp_new_token" },
    });
    fireEvent.click(screen.getByTestId("git-sync-save-button"));

    expect(mutate.mock.calls[0][0]).toEqual({ token: "ghp_new_token" });
  });

  it("shows the configured interval and sends a changed one", () => {
    render(
      <GitSyncConfigForm
        status={{ ...baseStatus, interval_seconds: 300 }}
        canManage
      />,
    );

    expect(screen.getByTestId("git-sync-interval-input")).toHaveValue(300);

    fireEvent.change(screen.getByTestId("git-sync-interval-input"), {
      target: { value: "60" },
    });
    fireEvent.click(screen.getByTestId("git-sync-save-button"));

    expect(mutate.mock.calls[0][0]).toEqual({ interval_seconds: 60 });
  });

  it("treats a blank interval as manual-only rather than no change", () => {
    // Clearing the field must mean 0 (manual), never "leave the timer on".
    render(
      <GitSyncConfigForm
        status={{ ...baseStatus, interval_seconds: 300 }}
        canManage
      />,
    );

    fireEvent.change(screen.getByTestId("git-sync-interval-input"), {
      target: { value: "" },
    });
    fireEvent.click(screen.getByTestId("git-sync-save-button"));

    expect(mutate.mock.calls[0][0]).toEqual({ interval_seconds: 0 });
  });

  it("clearing the token disables the text field and sends null", () => {
    render(<GitSyncConfigForm status={baseStatus} canManage />);

    fireEvent.click(screen.getByTestId("git-sync-clear-token-switch"));

    expect(screen.getByTestId("git-sync-token-input")).toBeDisabled();
    expect(screen.getByTestId("git-sync-save-button")).not.toBeDisabled();

    fireEvent.click(screen.getByTestId("git-sync-save-button"));

    expect(mutate.mock.calls[0][0]).toEqual({ token: null });
  });
});
