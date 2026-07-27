import type { ComponentProps } from "react";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { WorkspaceDropdown } from "../../../../src/components/features/home/workspace-dropdown/workspace-dropdown";
import type { LocalWorkspace, LocalWorkspaceParent } from "#/types/workspace";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

// Two parents whose NAMES deliberately differ from their path basenames, so a
// passing assertion proves the header uses the parent name (not basename-of-path).
const PARENTS: LocalWorkspaceParent[] = [
  { id: "p1", name: "Projects", path: "/projects" },
  { id: "p2", name: "My Work", path: "/work" },
];

const GROUPED_WORKSPACES: LocalWorkspace[] = [
  {
    id: "/projects/alpha",
    name: "alpha",
    path: "/projects/alpha",
    parentPath: "/projects",
  },
  { id: "/work/beta", name: "beta", path: "/work/beta", parentPath: "/work" },
  {
    id: "/projects/gamma",
    name: "gamma",
    path: "/projects/gamma",
    parentPath: "/projects",
  },
];

function renderDropdown(
  props: Partial<ComponentProps<typeof WorkspaceDropdown>> = {},
) {
  const onChange = vi.fn();
  render(
    <WorkspaceDropdown
      workspaces={GROUPED_WORKSPACES}
      parents={PARENTS}
      value={null}
      onChange={onChange}
      onAddClick={vi.fn()}
      onManageClick={vi.fn()}
      {...props}
    />,
  );
  return { onChange };
}

async function openMenu(user: ReturnType<typeof userEvent.setup>) {
  const input = screen.getByTestId("workspace-dropdown");
  await user.click(input);
  return screen.findByTestId("workspace-dropdown-menu");
}

describe("WorkspaceDropdown grouping (#129)", () => {
  it("renders a header with the parent's NAME (not the path basename) per group", async () => {
    const user = userEvent.setup();
    renderDropdown();
    const menu = await openMenu(user);

    const headers = within(menu).getAllByTestId("workspace-group-header");
    const headerText = headers.map((h) => h.textContent);
    expect(headerText).toContain("Projects"); // not "projects"
    expect(headerText).toContain("My Work"); // not "work"
  });

  it("groups every folder under its parent and shows all folders", async () => {
    const user = userEvent.setup();
    renderDropdown();
    const menu = await openMenu(user);

    expect(within(menu).getByText("alpha")).toBeInTheDocument();
    expect(within(menu).getByText("beta")).toBeInTheDocument();
    expect(within(menu).getByText("gamma")).toBeInTheDocument();
    // alpha and gamma (both /projects) render contiguously, before beta's group
    // boundary is crossed — exactly one header per distinct parent.
    expect(within(menu).getAllByTestId("workspace-group-header")).toHaveLength(
      2,
    );
  });

  it("keyboard navigation skips headers: ArrowDown+Enter selects a real folder", async () => {
    const user = userEvent.setup();
    const { onChange } = renderDropdown();
    await openMenu(user);

    await user.keyboard("{ArrowDown}{Enter}");

    // A header must never be selectable; the first highlight lands on a workspace.
    expect(onChange).toHaveBeenCalledTimes(1);
    const selected = onChange.mock.calls[0][0] as LocalWorkspace | null;
    expect(selected).not.toBeNull();
    expect(GROUPED_WORKSPACES.map((w) => w.id)).toContain(selected?.id);
  });

  it("does NOT render headers when there is only one group (flat fallback)", async () => {
    const user = userEvent.setup();
    renderDropdown({
      workspaces: [
        {
          id: "/projects/alpha",
          name: "alpha",
          path: "/projects/alpha",
          parentPath: "/projects",
        },
        {
          id: "/projects/gamma",
          name: "gamma",
          path: "/projects/gamma",
          parentPath: "/projects",
        },
      ],
    });
    const menu = await openMenu(user);

    expect(within(menu).queryByTestId("workspace-group-header")).toBeNull();
    expect(within(menu).getByText("alpha")).toBeInTheDocument();
    expect(within(menu).getByText("gamma")).toBeInTheDocument();
  });

  it("groups static (no-parent) workspaces under their own header", async () => {
    const user = userEvent.setup();
    renderDropdown({
      workspaces: [
        {
          id: "/projects/alpha",
          name: "alpha",
          path: "/projects/alpha",
          parentPath: "/projects",
        },
        { id: "/standalone", name: "standalone", path: "/standalone" }, // no parentPath
      ],
    });
    const menu = await openMenu(user);

    const headers = within(menu).getAllByTestId("workspace-group-header");
    // one header for the named parent, one for the static group
    expect(headers).toHaveLength(2);
    expect(within(menu).getByText("Projects")).toBeInTheDocument();
    expect(within(menu).getByText("standalone")).toBeInTheDocument();
  });

  it("labels the static group with the HOME$WORKSPACE_GROUP_OTHER i18n key", async () => {
    const user = userEvent.setup();
    renderDropdown({
      workspaces: [
        {
          id: "/projects/alpha",
          name: "alpha",
          path: "/projects/alpha",
          parentPath: "/projects",
        },
        { id: "/standalone", name: "standalone", path: "/standalone" },
      ],
    });
    const menu = await openMenu(user);

    // `t` is mocked to echo the key, so this pins the i18n wiring.
    expect(
      within(menu).getByText("HOME$WORKSPACE_GROUP_OTHER"),
    ).toBeInTheDocument();
  });

  it("renders the static 'Other' group last, even when a standalone appears first", async () => {
    const user = userEvent.setup();
    renderDropdown({
      workspaces: [
        { id: "/loose", name: "loose", path: "/loose" }, // standalone, appears FIRST
        {
          id: "/work/beta",
          name: "beta",
          path: "/work/beta",
          parentPath: "/work",
        },
      ],
      parents: [{ id: "p2", name: "My Work", path: "/work" }],
    });
    const menu = await openMenu(user);

    const headerText = within(menu)
      .getAllByTestId("workspace-group-header")
      .map((h) => h.textContent);
    expect(headerText).toEqual(["My Work", "HOME$WORKSPACE_GROUP_OTHER"]);
  });

  it("falls back to the path basename when a folder's parent is absent from `parents`", async () => {
    const user = userEvent.setup();
    renderDropdown({
      workspaces: [
        {
          id: "/work/beta",
          name: "beta",
          path: "/work/beta",
          parentPath: "/work",
        },
        {
          id: "/unknown/x",
          name: "x",
          path: "/unknown/x",
          parentPath: "/unknown",
        },
      ],
      parents: [{ id: "p2", name: "My Work", path: "/work" }], // /unknown intentionally absent
    });
    const menu = await openMenu(user);

    const headerText = within(menu)
      .getAllByTestId("workspace-group-header")
      .map((h) => h.textContent);
    expect(headerText).toContain("My Work");
    expect(headerText).toContain("unknown"); // basename of /unknown
  });

  it("folds the group label into each option's accessible name (a11y)", async () => {
    const user = userEvent.setup();
    renderDropdown();
    const menu = await openMenu(user);

    // The visual header is presentational; the group lives in each option's name.
    expect(
      within(menu).getByRole("option", { name: "Projects, alpha" }),
    ).toBeInTheDocument();
    expect(
      within(menu).getByRole("option", { name: "My Work, beta" }),
    ).toBeInTheDocument();
  });

  it("stays flat (no headers) for an all-standalone list with no parents", async () => {
    const user = userEvent.setup();
    renderDropdown({
      parents: undefined, // exercises the single STATIC_GROUP_KEY group path
      workspaces: [
        { id: "/a", name: "a", path: "/a" },
        { id: "/b", name: "b", path: "/b" },
      ],
    });
    const menu = await openMenu(user);

    expect(within(menu).queryByTestId("workspace-group-header")).toBeNull();
    expect(within(menu).getByText("a")).toBeInTheDocument();
    expect(within(menu).getByText("b")).toBeInTheDocument();
  });

  it("does not set an aria-label on options in flat (ungrouped) mode", async () => {
    const user = userEvent.setup();
    renderDropdown({
      parents: undefined,
      workspaces: [
        { id: "/a", name: "a", path: "/a" },
        { id: "/b", name: "b", path: "/b" },
      ],
    });
    const menu = await openMenu(user);

    // No grouping → the option's accessible name is just its display text.
    const option = within(menu).getByRole("option", { name: "a" });
    expect(option).not.toHaveAttribute("aria-label");
  });
});
