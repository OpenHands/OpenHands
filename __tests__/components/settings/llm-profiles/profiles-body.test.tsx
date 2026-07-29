import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ProfilesBody } from "#/components/features/settings/llm-profiles/profiles-body";
import { ProfileInfo } from "#/api/profiles-service/profiles-service.api";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => {
      const translations: Record<string, string> = {
        SETTINGS$PROFILES_LOAD_ERROR: "Failed to load profiles",
        SETTINGS$PROFILES_EMPTY: "No profiles saved yet",
        SETTINGS$PROFILE_ACTIVE: "Active",
        SETTINGS$PROFILE_MENU: "Profile menu",
        SETTINGS$PROFILE_EDIT: "Edit",
        BUTTON$RENAME: "Rename",
        SETTINGS$PROFILE_SET_ACTIVE: "Set as active",
        BUTTON$DELETE: "Delete",
      };
      return translations[key] || key;
    },
  }),
}));

const mockProfiles: ProfileInfo[] = [
  {
    name: "gpt-4-profile",
    model: "openai/gpt-4",
    base_url: null,
    api_key_set: true,
  },
  {
    name: "claude-profile",
    model: "anthropic/claude-3",
    base_url: "https://api.anthropic.com",
    api_key_set: false,
  },
];

const defaultProps = {
  isLoading: false,
  loadError: null,
  profiles: mockProfiles,
  active: "gpt-4-profile",
  canManage: true,
  onActivate: vi.fn(),
  onEdit: vi.fn(),
  onRename: vi.fn(),
  onDuplicate: vi.fn(),
  onDelete: vi.fn(),
  isActivating: false,
};

describe("ProfilesBody", () => {
  it("shows loading spinner when isLoading is true", () => {
    render(<ProfilesBody {...defaultProps} isLoading profiles={[]} />);

    expect(screen.getByTestId("loading-spinner")).toBeInTheDocument();
  });

  it("shows error message when loadError is present", () => {
    render(
      <ProfilesBody
        {...defaultProps}
        loadError={new Error("Network error")}
        profiles={[]}
      />,
    );

    expect(screen.getByText("Failed to load profiles")).toBeInTheDocument();
  });

  it("shows empty state when profiles array is empty", () => {
    render(<ProfilesBody {...defaultProps} profiles={[]} />);

    expect(screen.getByText("No profiles saved yet")).toBeInTheDocument();
  });

  it("renders a list of profiles", () => {
    render(<ProfilesBody {...defaultProps} />);

    expect(screen.getByText("gpt-4-profile")).toBeInTheDocument();
    expect(screen.getByText("openai/gpt-4")).toBeInTheDocument();
    expect(screen.getByText("claude-profile")).toBeInTheDocument();
    expect(screen.getByText("anthropic/claude-3")).toBeInTheDocument();
  });

  it("renders profile rows for each profile", () => {
    render(<ProfilesBody {...defaultProps} />);

    const rows = screen.getAllByTestId("profile-row");
    expect(rows).toHaveLength(2);
  });

  it("shows Active badge for the active profile", () => {
    render(<ProfilesBody {...defaultProps} active="gpt-4-profile" />);

    // Only the active profile should have the badge
    const badges = screen.getAllByTestId("profile-active-badge");
    expect(badges).toHaveLength(1);
  });

  it("loading state takes priority over error", () => {
    render(
      <ProfilesBody
        {...defaultProps}
        isLoading
        loadError={new Error("Error")}
        profiles={[]}
      />,
    );

    expect(screen.getByTestId("loading-spinner")).toBeInTheDocument();
    expect(
      screen.queryByText("Failed to load profiles"),
    ).not.toBeInTheDocument();
  });

  it("error state takes priority over empty state", () => {
    render(
      <ProfilesBody
        {...defaultProps}
        loadError={new Error("Error")}
        profiles={[]}
      />,
    );

    expect(screen.getByText("Failed to load profiles")).toBeInTheDocument();
    expect(screen.queryByText("No profiles saved yet")).not.toBeInTheDocument();
  });
});
