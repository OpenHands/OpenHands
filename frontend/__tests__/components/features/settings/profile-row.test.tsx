import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { LlmProfileSummary } from "#/api/settings-service/profiles-service.api";
import { ProfileRow } from "#/components/features/settings/profile-row";

const profile: LlmProfileSummary = {
  name: "openai_gpt-4o",
  model: "openai/gpt-4o",
  base_url: null,
  api_key_set: true,
};

function renderRow(
  overrides: Partial<React.ComponentProps<typeof ProfileRow>> = {},
) {
  const props = {
    profile,
    isActive: false,
    onActivate: vi.fn(),
    onEdit: vi.fn(),
    onRename: vi.fn(),
    onDelete: vi.fn(),
    isActivating: false,
    ...overrides,
  };
  return {
    // eslint-disable-next-line react/jsx-props-no-spreading
    ...render(<ProfileRow {...props} />),
    props,
  };
}

describe("ProfileRow", () => {
  it("renders the profile name and model", () => {
    renderRow();

    expect(screen.getByText("openai_gpt-4o")).toBeInTheDocument();
    expect(screen.getByText("openai/gpt-4o")).toBeInTheDocument();
    expect(
      screen.queryByTestId("profile-active-badge"),
    ).not.toBeInTheDocument();
  });

  it("shows the active badge when isActive is true", () => {
    renderRow({ isActive: true });

    expect(screen.getByTestId("profile-active-badge")).toHaveTextContent(
      "SETTINGS$PROFILE_ACTIVE_BADGE",
    );
  });

  it("opens the actions menu when the trigger is clicked", async () => {
    renderRow();
    const user = userEvent.setup();

    expect(screen.queryByTestId("profile-edit")).not.toBeInTheDocument();
    await user.click(screen.getByTestId("profile-menu-trigger"));
    expect(screen.getByTestId("profile-edit")).toBeInTheDocument();
  });
});
