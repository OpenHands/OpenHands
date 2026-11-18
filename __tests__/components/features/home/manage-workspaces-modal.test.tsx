import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ManageWorkspacesModal } from "#/components/features/home/workspace-dropdown/manage-workspaces-modal";
import { LocalWorkspace, LocalWorkspaceParent } from "#/types/workspace";

const { mockGetCachedAgentServerVersion } = vi.hoisted(() => ({
  mockGetCachedAgentServerVersion: vi.fn((): string | null => null),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

vi.mock("#/api/agent-server-compatibility", async () => {
  const actual = await vi.importActual<
    typeof import("#/api/agent-server-compatibility")
  >("#/api/agent-server-compatibility");
  return {
    ...actual,
    getCachedAgentServerVersion: () => mockGetCachedAgentServerVersion(),
  };
});

const WORKSPACE: LocalWorkspace = {
  id: "/Users/me/dev/repo1",
  name: "repo1",
  path: "/Users/me/dev/repo1",
};

const OTHER_WORKSPACE: LocalWorkspace = {
  id: "/Users/me/dev/repo2",
  name: "repo2",
  path: "/Users/me/dev/repo2",
};

const PARENT: LocalWorkspaceParent = {
  id: "/Users/me/dev",
  name: "dev",
  path: "/Users/me/dev",
};

const CHILD: LocalWorkspace = {
  id: "/Users/me/dev/repo3",
  name: "repo3",
  path: "/Users/me/dev/repo3",
  parentPath: "/Users/me/dev",
};

function renderModal({
  workspaces = [WORKSPACE, OTHER_WORKSPACE],
  workspaceParents = [],
  onRename,
  onRenameParent,
}: {
  workspaces?: LocalWorkspace[];
  workspaceParents?: LocalWorkspaceParent[];
  onRename?: (path: string, name: string) => Promise<unknown> | void;
  onRenameParent?: (path: string, name: string) => Promise<unknown> | void;
} = {}) {
  render(
    <ManageWorkspacesModal
      isOpen
      workspaces={workspaces}
      workspaceParents={workspaceParents}
      onClose={vi.fn()}
      onRemove={vi.fn()}
      onRemoveParent={vi.fn()}
      onRename={onRename}
      onRenameParent={onRenameParent}
    />,
  );
}

beforeEach(() => {
  mockGetCachedAgentServerVersion.mockReturnValue(null);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("ManageWorkspacesModal rename", () => {
  it("offers rename for saved workspaces and parents but not parent-derived children", () => {
    renderModal({
      workspaces: [WORKSPACE, CHILD],
      workspaceParents: [PARENT],
      onRename: vi.fn(),
      onRenameParent: vi.fn(),
    });

    expect(screen.getByTestId("manage-workspaces-rename-repo1")).toBeEnabled();
    expect(
      screen.getByTestId("manage-workspaces-rename-parent-dev"),
    ).toBeEnabled();
    expect(screen.queryByTestId("manage-workspaces-rename-repo3")).toBeNull();
  });

  it("hides the rename control when the consumer passes no handler", () => {
    renderModal();

    expect(screen.queryByTestId("manage-workspaces-rename-repo1")).toBeNull();
  });

  it("opens an inline editor seeded with the current name and saves the trimmed value", async () => {
    const onRename = vi.fn().mockResolvedValue(undefined);
    renderModal({ onRename });
    const user = userEvent.setup();

    await user.click(screen.getByTestId("manage-workspaces-rename-repo1"));

    const input = screen.getByTestId(
      "manage-workspaces-rename-workspace-input",
    );
    expect(input).toHaveValue("repo1");

    await user.clear(input);
    await user.type(input, "  Renamed repo  ");
    await user.click(
      screen.getByTestId("manage-workspaces-rename-workspace-save"),
    );

    await waitFor(() =>
      expect(onRename).toHaveBeenCalledWith(WORKSPACE.path, "Renamed repo"),
    );
    await waitFor(() =>
      expect(
        screen.queryByTestId("manage-workspaces-rename-workspace-input"),
      ).toBeNull(),
    );
  });

  it("saves on Enter", async () => {
    const onRename = vi.fn().mockResolvedValue(undefined);
    renderModal({ onRename });
    const user = userEvent.setup();

    await user.click(screen.getByTestId("manage-workspaces-rename-repo1"));
    const input = screen.getByTestId(
      "manage-workspaces-rename-workspace-input",
    );
    await user.clear(input);
    await user.type(input, "renamed{Enter}");

    await waitFor(() =>
      expect(onRename).toHaveBeenCalledWith(WORKSPACE.path, "renamed"),
    );
  });

  it("cancels on Escape without dispatching", async () => {
    const onRename = vi.fn().mockResolvedValue(undefined);
    renderModal({ onRename });
    const user = userEvent.setup();

    await user.click(screen.getByTestId("manage-workspaces-rename-repo1"));
    const input = screen.getByTestId(
      "manage-workspaces-rename-workspace-input",
    );
    await user.clear(input);
    await user.type(input, "discarded{Escape}");

    await waitFor(() =>
      expect(
        screen.queryByTestId("manage-workspaces-rename-workspace-input"),
      ).toBeNull(),
    );
    expect(onRename).not.toHaveBeenCalled();
    expect(screen.getByText("repo1")).toBeInTheDocument();
  });

  it("disables saving for an empty, unchanged, or over-long name", async () => {
    const onRename = vi.fn().mockResolvedValue(undefined);
    renderModal({ onRename });
    const user = userEvent.setup();

    await user.click(screen.getByTestId("manage-workspaces-rename-repo1"));
    const input = screen.getByTestId(
      "manage-workspaces-rename-workspace-input",
    );
    const save = screen.getByTestId("manage-workspaces-rename-workspace-save");

    expect(save).toBeDisabled();

    await user.clear(input);
    expect(save).toBeDisabled();
    expect(screen.getByText("HOME$WORKSPACE_NAME_RULE")).toBeInTheDocument();

    await user.type(input, "   ");
    expect(save).toBeDisabled();

    await user.clear(input);
    await user.type(input, "renamed");
    expect(save).toBeEnabled();
  });

  it("does not commit on blur", async () => {
    const onRename = vi.fn().mockResolvedValue(undefined);
    renderModal({ onRename });
    const user = userEvent.setup();

    await user.click(screen.getByTestId("manage-workspaces-rename-repo1"));
    const input = screen.getByTestId(
      "manage-workspaces-rename-workspace-input",
    );
    await user.clear(input);
    await user.type(input, "renamed");
    await user.click(screen.getByTestId("manage-workspaces-done"));

    expect(onRename).not.toHaveBeenCalled();
    expect(input).toHaveValue("renamed");
  });

  it("keeps the editor open with the typed text when the rename fails", async () => {
    const onRename = vi.fn().mockRejectedValue(new Error("boom"));
    renderModal({ onRename });
    const user = userEvent.setup();

    await user.click(screen.getByTestId("manage-workspaces-rename-repo1"));
    const input = screen.getByTestId(
      "manage-workspaces-rename-workspace-input",
    );
    await user.clear(input);
    await user.type(input, "renamed");
    await user.click(
      screen.getByTestId("manage-workspaces-rename-workspace-save"),
    );

    await waitFor(() => expect(onRename).toHaveBeenCalledTimes(1));
    expect(
      screen.getByTestId("manage-workspaces-rename-workspace-input"),
    ).toHaveValue("renamed");
  });

  it("edits one row at a time and renames parents through their own handler", async () => {
    const onRename = vi.fn().mockResolvedValue(undefined);
    const onRenameParent = vi.fn().mockResolvedValue(undefined);
    renderModal({
      workspaces: [WORKSPACE, OTHER_WORKSPACE],
      workspaceParents: [PARENT],
      onRename,
      onRenameParent,
    });
    const user = userEvent.setup();

    await user.click(screen.getByTestId("manage-workspaces-rename-repo1"));
    await user.click(screen.getByTestId("manage-workspaces-rename-parent-dev"));

    expect(
      screen.getAllByTestId(
        /manage-workspaces-rename-(workspace|parent)-input/,
      ),
    ).toHaveLength(1);

    const input = screen.getByTestId("manage-workspaces-rename-parent-input");
    await user.clear(input);
    await user.type(input, "development{Enter}");

    await waitFor(() =>
      expect(onRenameParent).toHaveBeenCalledWith(PARENT.path, "development"),
    );
    expect(onRename).not.toHaveBeenCalled();
  });

  it("disables rename against a backend older than the feature", () => {
    mockGetCachedAgentServerVersion.mockReturnValue("1.62.0");
    renderModal({
      workspaceParents: [PARENT],
      onRename: vi.fn(),
      onRenameParent: vi.fn(),
    });

    expect(screen.getByTestId("manage-workspaces-rename-repo1")).toBeDisabled();
    expect(
      screen.getByTestId("manage-workspaces-rename-parent-dev"),
    ).toBeDisabled();
  });
});
