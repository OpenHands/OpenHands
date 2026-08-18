import { render, screen, fireEvent, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { I18nKey } from "#/i18n/declaration";
import type { GitSyncStatus } from "#/types/git-sync";
import { GitSyncConfigForm } from "./git-sync-config-form";

const mutate = vi.fn();
const checkConfig = vi.fn();
const syncNow = vi.fn();

vi.mock("#/hooks/query/use-git-sync", () => ({
  useUpdateGitSyncConfig: () => ({ mutate, isPending: false }),
  useCheckGitSyncConfig: () => ({
    mutateAsync: checkConfig,
    isPending: false,
  }),
}));

/** Submitting is async now: it awaits the reachability check before saving. */
const clickSave = async () => {
  await act(async () => {
    fireEvent.click(screen.getByTestId("git-sync-save-button"));
  });
};

beforeEach(() => {
  // Reachable unless a test says otherwise, so the save path stays the
  // default one under test.
  checkConfig.mockResolvedValue({
    ok: true,
    branch_exists: true,
    detail: null,
  });
});

const displayErrorToast = vi.fn();
const displaySuccessToast = vi.fn();

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: (message: string) => displayErrorToast(message),
  displaySuccessToast: (message: string) => displaySuccessToast(message),
}));

/** Make the next save resolve down the success or the failure path. */
const respondWith = (outcome: "success" | { error: unknown }) => {
  mutate.mockImplementation((_body, options) => {
    if (outcome === "success") options.onSuccess?.();
    else options.onError?.(outcome.error);
    // react-query runs onSettled on both paths.
    options.onSettled?.();
  });
};

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
  it("always renders the token and encryption key fields empty", async () => {
    render(
      <GitSyncConfigForm status={baseStatus} canManage onSyncNow={syncNow} />,
    );

    expect(screen.getByTestId("git-sync-token-input")).toHaveValue("");
    expect(screen.getByTestId("git-sync-encryption-key-input")).toHaveValue("");
  });

  it("shows the encryption key placeholder based on encryption_enabled", async () => {
    const { rerender } = render(
      <GitSyncConfigForm status={baseStatus} canManage onSyncNow={syncNow} />,
    );
    expect(screen.getByTestId("git-sync-encryption-key-input")).toHaveAttribute(
      "placeholder",
      I18nKey.AUTOMATIONS$GIT_SYNC$KEY_UNSET_PLACEHOLDER,
    );

    rerender(
      <GitSyncConfigForm
        status={{ ...baseStatus, encryption_enabled: true }}
        canManage
        onSyncNow={syncNow}
      />,
    );
    expect(screen.getByTestId("git-sync-encryption-key-input")).toHaveAttribute(
      "placeholder",
      I18nKey.AUTOMATIONS$GIT_SYNC$KEY_SET_PLACEHOLDER,
    );
  });

  it("enables submit once a field changes and sends only the changed field", async () => {
    render(
      <GitSyncConfigForm status={baseStatus} canManage onSyncNow={syncNow} />,
    );

    expect(screen.getByTestId("git-sync-save-button")).toBeDisabled();

    fireEvent.change(screen.getByTestId("git-sync-branch-input"), {
      target: { value: "develop" },
    });
    expect(screen.getByTestId("git-sync-save-button")).not.toBeDisabled();

    await clickSave();

    expect(mutate).toHaveBeenCalledTimes(1);
    expect(mutate.mock.calls[0][0]).toEqual({ branch: "develop" });
  });

  it("sends a typed token as a plain string, omitting unrelated fields", async () => {
    render(
      <GitSyncConfigForm status={baseStatus} canManage onSyncNow={syncNow} />,
    );

    fireEvent.change(screen.getByTestId("git-sync-token-input"), {
      target: { value: "ghp_new_token" },
    });
    await clickSave();

    expect(mutate.mock.calls[0][0]).toEqual({ token: "ghp_new_token" });
  });

  it("shows the configured interval and sends a changed one", async () => {
    render(
      <GitSyncConfigForm
        status={{ ...baseStatus, interval_seconds: 300 }}
        canManage
        onSyncNow={syncNow}
      />,
    );

    expect(screen.getByTestId("git-sync-interval-input")).toHaveValue(300);

    fireEvent.change(screen.getByTestId("git-sync-interval-input"), {
      target: { value: "60" },
    });
    await clickSave();

    expect(mutate.mock.calls[0][0]).toEqual({ interval_seconds: 60 });
  });

  it("treats a blank interval as manual-only rather than no change", async () => {
    // Clearing the field must mean 0 (manual), never "leave the timer on".
    render(
      <GitSyncConfigForm
        status={{ ...baseStatus, interval_seconds: 300 }}
        canManage
        onSyncNow={syncNow}
      />,
    );

    fireEvent.change(screen.getByTestId("git-sync-interval-input"), {
      target: { value: "" },
    });
    await clickSave();

    expect(mutate.mock.calls[0][0]).toEqual({ interval_seconds: 0 });
  });

  it("clearing the token disables the text field and sends null", async () => {
    render(
      <GitSyncConfigForm status={baseStatus} canManage onSyncNow={syncNow} />,
    );

    fireEvent.click(screen.getByTestId("git-sync-clear-token-switch"));

    expect(screen.getByTestId("git-sync-token-input")).toBeDisabled();
    expect(screen.getByTestId("git-sync-save-button")).not.toBeDisabled();

    await clickSave();

    expect(mutate.mock.calls[0][0]).toEqual({ token: null });
  });

  // Regression: these posted "" rather than null, and the backend stored the
  // empty string as a literal override. An empty branch or path then made the
  // next git command fatal (`git checkout -B ""`, `git add -A -- ""`), wedging
  // every subsequent sync cycle.
  it.each([
    ["git-sync-branch-input", "branch"],
    ["git-sync-path-input", "path"],
    ["git-sync-repo-url-input", "repo_url"],
  ])("clearing %s sends null, not an empty string", async (testId, field) => {
    render(
      <GitSyncConfigForm status={baseStatus} canManage onSyncNow={syncNow} />,
    );

    fireEvent.change(screen.getByTestId(testId), { target: { value: "" } });
    await clickSave();

    expect(mutate.mock.calls[0][0]).toEqual({ [field]: null });
  });

  it("sends a whitespace-only field as null", async () => {
    render(
      <GitSyncConfigForm status={baseStatus} canManage onSyncNow={syncNow} />,
    );

    fireEvent.change(screen.getByTestId("git-sync-path-input"), {
      target: { value: "   " },
    });
    await clickSave();

    expect(mutate.mock.calls[0][0]).toEqual({ path: null });
  });

  // Regression: the clear switch remounts the secret input, which emptied the
  // DOM field but left the typed value in state -- so a change of mind sent
  // `token: ""`, an override that fails every later push with
  // `fatal: Authentication failed`.
  it.each([
    ["git-sync-token-input", "git-sync-clear-token-switch"],
    ["git-sync-encryption-key-input", "git-sync-clear-encryption-key-switch"],
  ])(
    "forgets a typed %s when its clear switch is toggled on and back off",
    async (inputTestId, switchTestId) => {
      render(
        <GitSyncConfigForm status={baseStatus} canManage onSyncNow={syncNow} />,
      );

      fireEvent.change(screen.getByTestId(inputTestId), {
        target: { value: "s3cret" },
      });
      fireEvent.click(screen.getByTestId(switchTestId));
      fireEvent.click(screen.getByTestId(switchTestId));

      // Nothing left to save: the field is empty again and no secret is
      // pending, so an empty override can't be submitted.
      expect(screen.getByTestId(inputTestId)).toHaveValue("");
      expect(screen.getByTestId("git-sync-save-button")).toBeDisabled();

      fireEvent.change(screen.getByTestId("git-sync-branch-input"), {
        target: { value: "develop" },
      });
      await clickSave();

      expect(mutate.mock.calls[0][0]).toEqual({ branch: "develop" });
    },
  );

  it("keeps the edits and the enabled Save button when the save fails", async () => {
    // React resets an uncontrolled `<form action={...}>` as soon as the action
    // returns, which used to wipe the edits mid-flight; clearing the change
    // flags on `onSettled` then disabled Save, leaving nothing to retry with.
    respondWith({ error: { status: 500 } });
    render(
      <GitSyncConfigForm status={baseStatus} canManage onSyncNow={syncNow} />,
    );

    fireEvent.change(screen.getByTestId("git-sync-branch-input"), {
      target: { value: "develop" },
    });
    fireEvent.click(screen.getByTestId("git-sync-clear-token-switch"));
    await clickSave();

    expect(screen.getByTestId("git-sync-branch-input")).toHaveValue("develop");
    expect(screen.getByTestId("git-sync-save-button")).not.toBeDisabled();
    expect(screen.getByTestId("git-sync-clear-token-switch")).toBeChecked();
  });

  it("goes clean again after a successful save", async () => {
    respondWith("success");
    render(
      <GitSyncConfigForm status={baseStatus} canManage onSyncNow={syncNow} />,
    );

    fireEvent.click(screen.getByTestId("git-sync-clear-token-switch"));
    await clickSave();

    expect(screen.getByTestId("git-sync-save-button")).toBeDisabled();
    expect(screen.getByTestId("git-sync-clear-token-switch")).not.toBeChecked();
    expect(displaySuccessToast).toHaveBeenCalledWith(
      I18nKey.AUTOMATIONS$GIT_SYNC$CONFIG_SAVED,
    );
  });

  // Regression: the author fields were dirty only while non-empty, so emptying
  // one never posted `author_name: null` and a wrong override was permanent.
  it.each([
    ["git-sync-author-name-input", "author_name"],
    ["git-sync-author-email-input", "author_email"],
  ])(
    "clearing %s resets the override to the default",
    async (testId, field) => {
      render(
        <GitSyncConfigForm status={baseStatus} canManage onSyncNow={syncNow} />,
      );

      fireEvent.change(screen.getByTestId(testId), {
        target: { value: "someone@example.com" },
      });
      fireEvent.change(screen.getByTestId(testId), { target: { value: "" } });
      await clickSave();

      expect(mutate.mock.calls[0][0]).toEqual({ [field]: null });
    },
  );

  it("sends the enabled flag when the sync switch is flipped", async () => {
    render(
      <GitSyncConfigForm status={baseStatus} canManage onSyncNow={syncNow} />,
    );

    expect(screen.getByTestId("git-sync-enabled-switch")).toBeChecked();

    fireEvent.click(screen.getByTestId("git-sync-enabled-switch"));
    await clickSave();

    expect(mutate.mock.calls[0][0]).toEqual({ enabled: false });
  });

  it("leaves enabled out of the request when the switch is not touched", async () => {
    render(
      <GitSyncConfigForm status={baseStatus} canManage onSyncNow={syncNow} />,
    );

    fireEvent.change(screen.getByTestId("git-sync-branch-input"), {
      target: { value: "develop" },
    });
    await clickSave();

    expect(mutate.mock.calls[0][0]).toEqual({ branch: "develop" });
  });

  it("explains the restart requirement when the backend refuses to enable sync", async () => {
    // The backend answers 409 when it booted without git sync turned on --
    // the raw detail is a wall of text, and the generic error message would
    // read as a transient failure worth retrying.
    respondWith({ error: { status: 409 } });
    render(
      <GitSyncConfigForm
        status={{ ...baseStatus, enabled: false }}
        canManage
        onSyncNow={syncNow}
      />,
    );

    fireEvent.click(screen.getByTestId("git-sync-enabled-switch"));
    await clickSave();

    expect(displayErrorToast).toHaveBeenCalledWith(
      I18nKey.AUTOMATIONS$GIT_SYNC$ENABLE_BLOCKED_ERROR,
    );
  });

  describe("reachability check", () => {
    const unreachable = () =>
      checkConfig.mockResolvedValue({
        ok: false,
        branch_exists: false,
        detail: "fatal: repository not found",
      });

    it("tests the settings against the remote before saving them", async () => {
      render(
        <GitSyncConfigForm status={baseStatus} canManage onSyncNow={syncNow} />,
      );

      fireEvent.change(screen.getByTestId("git-sync-repo-url-input"), {
        target: { value: "https://example.com/org/other.git" },
      });
      await clickSave();

      // The same body that is about to be saved, so the check answers for
      // the configuration that would actually take effect.
      expect(checkConfig).toHaveBeenCalledWith({
        repo_url: "https://example.com/org/other.git",
      });
      expect(mutate).toHaveBeenCalledTimes(1);
    });

    it("does not save a configuration that cannot reach its repo", async () => {
      unreachable();
      render(
        <GitSyncConfigForm status={baseStatus} canManage onSyncNow={syncNow} />,
      );

      fireEvent.change(screen.getByTestId("git-sync-repo-url-input"), {
        target: { value: "https://example.com/typo.git" },
      });
      await clickSave();

      expect(mutate).not.toHaveBeenCalled();
      expect(screen.getByTestId("git-sync-check-failure")).toHaveTextContent(
        "fatal: repository not found",
      );
      // Still dirty, so the operator can fix the field and try again.
      expect(screen.getByTestId("git-sync-save-button")).not.toBeDisabled();
    });

    it("saves the same values anyway when Save is pressed a second time", async () => {
      // A check is evidence, not an authority: a remote that refuses
      // `ls-remote` but accepts a push must not lock its operator out.
      unreachable();
      respondWith("success");
      render(
        <GitSyncConfigForm status={baseStatus} canManage onSyncNow={syncNow} />,
      );

      fireEvent.change(screen.getByTestId("git-sync-repo-url-input"), {
        target: { value: "https://example.com/typo.git" },
      });
      await clickSave();
      await clickSave();

      expect(checkConfig).toHaveBeenCalledTimes(1);
      expect(mutate).toHaveBeenCalledTimes(1);
      expect(mutate.mock.calls[0][0]).toEqual({
        repo_url: "https://example.com/typo.git",
      });
    });

    it("re-checks once the settings are edited after a failure", async () => {
      unreachable();
      render(
        <GitSyncConfigForm status={baseStatus} canManage onSyncNow={syncNow} />,
      );

      fireEvent.change(screen.getByTestId("git-sync-repo-url-input"), {
        target: { value: "https://example.com/typo.git" },
      });
      await clickSave();

      checkConfig.mockResolvedValue({
        ok: true,
        branch_exists: false,
        detail: null,
      });
      fireEvent.change(screen.getByTestId("git-sync-repo-url-input"), {
        target: { value: "https://example.com/fixed.git" },
      });
      await clickSave();

      expect(checkConfig).toHaveBeenCalledTimes(2);
      expect(mutate.mock.calls[0][0]).toEqual({
        repo_url: "https://example.com/fixed.git",
      });
      expect(screen.queryByTestId("git-sync-check-failure")).toBeNull();
    });

    it("skips the check when nothing the remote could reject changed", async () => {
      render(
        <GitSyncConfigForm status={baseStatus} canManage onSyncNow={syncNow} />,
      );

      fireEvent.change(screen.getByTestId("git-sync-interval-input"), {
        target: { value: "60" },
      });
      await clickSave();

      expect(checkConfig).not.toHaveBeenCalled();
      expect(mutate).toHaveBeenCalledTimes(1);
    });

    it("saves anyway when the check itself cannot run", async () => {
      // An automation backend that predates the endpoint answers 404. A check
      // that says nothing must never be what blocks a save.
      checkConfig.mockRejectedValue({ status: 404 });
      render(
        <GitSyncConfigForm status={baseStatus} canManage onSyncNow={syncNow} />,
      );

      fireEvent.change(screen.getByTestId("git-sync-branch-input"), {
        target: { value: "develop" },
      });
      await clickSave();

      expect(mutate).toHaveBeenCalledTimes(1);
      expect(screen.queryByTestId("git-sync-check-failure")).toBeNull();
    });
  });

  describe("save and sync", () => {
    const clickSaveAndSync = async () => {
      await act(async () => {
        fireEvent.click(screen.getByTestId("git-sync-save-and-sync-button"));
      });
    };

    it("runs a cycle once the settings are stored", async () => {
      respondWith("success");
      render(
        <GitSyncConfigForm status={baseStatus} canManage onSyncNow={syncNow} />,
      );

      fireEvent.change(screen.getByTestId("git-sync-branch-input"), {
        target: { value: "develop" },
      });
      await clickSaveAndSync();

      expect(mutate.mock.calls[0][0]).toEqual({ branch: "develop" });
      expect(syncNow).toHaveBeenCalledTimes(1);
    });

    // The cycle runs off the stored configuration, so syncing a rejected save
    // would push the settings the operator was trying to replace.
    it("does not sync when the save is rejected", async () => {
      respondWith({ error: { status: 500 } });
      render(
        <GitSyncConfigForm status={baseStatus} canManage onSyncNow={syncNow} />,
      );

      fireEvent.change(screen.getByTestId("git-sync-branch-input"), {
        target: { value: "develop" },
      });
      await clickSaveAndSync();

      expect(syncNow).not.toHaveBeenCalled();
    });

    it("does not sync when the settings fail their check", async () => {
      checkConfig.mockResolvedValue({
        ok: false,
        branch_exists: false,
        detail: "repository not found",
      });
      render(
        <GitSyncConfigForm status={baseStatus} canManage onSyncNow={syncNow} />,
      );

      fireEvent.change(screen.getByTestId("git-sync-repo-url-input"), {
        target: { value: "https://example.com/typo.git" },
      });
      await clickSaveAndSync();

      expect(mutate).not.toHaveBeenCalled();
      expect(syncNow).not.toHaveBeenCalled();
    });

    it("does not sync a plain save", async () => {
      respondWith("success");
      render(
        <GitSyncConfigForm status={baseStatus} canManage onSyncNow={syncNow} />,
      );

      fireEvent.change(screen.getByTestId("git-sync-branch-input"), {
        target: { value: "develop" },
      });
      await clickSave();

      expect(mutate).toHaveBeenCalledTimes(1);
      expect(syncNow).not.toHaveBeenCalled();
    });

    // Regression: the button used to arm a flag on click, and native
    // validation swallows the submit that click was meant to start -- so the
    // flag was still set when Save was pressed next, syncing off a button
    // that never asked for it.
    it("does not carry a swallowed submit over to the next save", async () => {
      respondWith("success");
      render(
        <GitSyncConfigForm status={baseStatus} canManage onSyncNow={syncNow} />,
      );

      // An invalid email fails the input's own constraint, so the click never
      // reaches a submit.
      fireEvent.change(screen.getByTestId("git-sync-author-email-input"), {
        target: { value: "not-an-email" },
      });
      await clickSaveAndSync();
      expect(mutate).not.toHaveBeenCalled();

      fireEvent.change(screen.getByTestId("git-sync-author-email-input"), {
        target: { value: "ops@example.com" },
      });
      await clickSave();

      expect(mutate).toHaveBeenCalledTimes(1);
      expect(syncNow).not.toHaveBeenCalled();
    });

    // A trigger against disabled sync is a 503, so there is nothing to offer
    // until the same save turns it on.
    it("stays disabled while sync is off and arms when it is switched on", async () => {
      render(
        <GitSyncConfigForm
          status={{ ...baseStatus, enabled: false }}
          canManage
          onSyncNow={syncNow}
        />,
      );

      fireEvent.change(screen.getByTestId("git-sync-branch-input"), {
        target: { value: "develop" },
      });
      expect(
        screen.getByTestId("git-sync-save-and-sync-button"),
      ).toBeDisabled();
      expect(screen.getByTestId("git-sync-save-button")).toBeEnabled();

      fireEvent.click(screen.getByTestId("git-sync-enabled-switch"));

      expect(screen.getByTestId("git-sync-save-and-sync-button")).toBeEnabled();
    });
  });
});
