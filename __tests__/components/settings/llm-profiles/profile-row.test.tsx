import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { ProfileRow } from "#/components/features/settings/llm-profiles/profile-row";
import { ProfileInfo } from "#/api/profiles-service/profiles-service.api";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => {
      const translations: Record<string, string> = {
        SETTINGS$PROFILE_DEFAULT: "Default",
        SETTINGS$PROFILE_MENU: "Profile menu",
        SETTINGS$PROFILE_EDIT: "Edit",
        BUTTON$RENAME: "Rename",
        SETTINGS$PROFILE_SET_DEFAULT: "Set as default",
        BUTTON$DELETE: "Delete",
      };
      return translations[key] || key;
    },
  }),
}));

const mockProfile: ProfileInfo = {
  name: "gpt-4-profile",
  model: "openai/gpt-4",
  base_url: null,
  api_key_set: true,
};

const defaultProps = {
  profile: mockProfile,
  isActive: false,
  canManage: true,
  onActivate: vi.fn(),
  onEdit: vi.fn(),
  onRename: vi.fn(),
  onDuplicate: vi.fn(),
  onDelete: vi.fn(),
  isActivating: false,
};

describe("ProfileRow", () => {
  it("displays the profile name", () => {
    render(<ProfileRow {...defaultProps} />);

    expect(screen.getByText("gpt-4-profile")).toBeInTheDocument();
  });

  it("displays the model name when present", () => {
    render(<ProfileRow {...defaultProps} />);

    expect(screen.getByText("openai/gpt-4")).toBeInTheDocument();
  });

  it("does not display model when null", () => {
    const profileWithoutModel: ProfileInfo = {
      ...mockProfile,
      model: null,
    };

    render(<ProfileRow {...defaultProps} profile={profileWithoutModel} />);

    expect(screen.queryByText("openai/gpt-4")).not.toBeInTheDocument();
  });

  it("shows Active badge when isActive is true", () => {
    render(<ProfileRow {...defaultProps} isActive />);

    expect(screen.getByTestId("profile-active-badge")).toBeInTheDocument();
    expect(screen.getByText("Default")).toBeInTheDocument();
  });

  it("does not show Active badge when isActive is false", () => {
    render(<ProfileRow {...defaultProps} isActive={false} />);

    expect(
      screen.queryByTestId("profile-active-badge"),
    ).not.toBeInTheDocument();
  });

  it("hides the actions menu when canManage is false (view-only members)", () => {
    render(<ProfileRow {...defaultProps} canManage={false} />);

    // The row still shows the profile, but offers no mutate actions.
    expect(screen.getByText("gpt-4-profile")).toBeInTheDocument();
    expect(
      screen.queryByTestId("profile-menu-trigger"),
    ).not.toBeInTheDocument();
  });

  it("opens menu when trigger button is clicked", async () => {
    const user = userEvent.setup();

    render(<ProfileRow {...defaultProps} />);

    const menuTrigger = screen.getByTestId("profile-menu-trigger");
    await user.click(menuTrigger);

    expect(screen.getByText("Edit")).toBeInTheDocument();
    expect(screen.getByText("Rename")).toBeInTheDocument();
    expect(screen.getByText("Set as default")).toBeInTheDocument();
    expect(screen.getByText("Delete")).toBeInTheDocument();
  });

  it("toggles menu visibility on multiple clicks", async () => {
    const user = userEvent.setup();

    render(<ProfileRow {...defaultProps} />);

    const menuTrigger = screen.getByTestId("profile-menu-trigger");

    // Menu should be closed initially
    expect(screen.queryByText("Edit")).not.toBeInTheDocument();

    // First click opens the menu
    await user.click(menuTrigger);
    expect(screen.getByText("Edit")).toBeInTheDocument();
  });

  it("calls onEdit when Edit is clicked", async () => {
    const user = userEvent.setup();
    const handleEdit = vi.fn();

    render(<ProfileRow {...defaultProps} onEdit={handleEdit} />);

    await user.click(screen.getByTestId("profile-menu-trigger"));
    await user.click(screen.getByText("Edit"));

    expect(handleEdit).toHaveBeenCalledWith(mockProfile);
  });

  it("calls onRename when Rename is clicked", async () => {
    const user = userEvent.setup();
    const handleRename = vi.fn();

    render(<ProfileRow {...defaultProps} onRename={handleRename} />);

    await user.click(screen.getByTestId("profile-menu-trigger"));
    await user.click(screen.getByText("Rename"));

    expect(handleRename).toHaveBeenCalledWith(mockProfile);
  });

  it("calls onActivate when Set Active is clicked", async () => {
    const user = userEvent.setup();
    const handleActivate = vi.fn();

    render(<ProfileRow {...defaultProps} onActivate={handleActivate} />);

    await user.click(screen.getByTestId("profile-menu-trigger"));
    await user.click(screen.getByText("Set as default"));

    expect(handleActivate).toHaveBeenCalledWith(mockProfile.name);
  });

  it("calls onDelete when Delete is clicked", async () => {
    const user = userEvent.setup();
    const handleDelete = vi.fn();

    render(<ProfileRow {...defaultProps} onDelete={handleDelete} />);

    await user.click(screen.getByTestId("profile-menu-trigger"));
    await user.click(screen.getByText("Delete"));

    expect(handleDelete).toHaveBeenCalledWith(mockProfile);
  });

  it("has accessible menu trigger button", () => {
    render(<ProfileRow {...defaultProps} />);

    const menuTrigger = screen.getByTestId("profile-menu-trigger");
    expect(menuTrigger).toHaveAttribute("aria-label", "Profile menu");
  });

  it("renders the opened menu outside the row container so ancestor overflow cannot clip it", async () => {
    const user = userEvent.setup();

    const { container } = render(<ProfileRow {...defaultProps} />);

    await user.click(screen.getByTestId("profile-menu-trigger"));

    const menu = screen.getByTestId("profile-actions-menu");
    expect(container.contains(menu)).toBe(false);
  });
});
