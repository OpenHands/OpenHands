import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router";
import { SwitchProfileContextMenu } from "#/components/features/chat/switch-profile-context-menu";
import type { LlmProfileSummary } from "#/api/settings-service/profiles-service.api";

// `useTranslation` is mocked globally in vitest.setup.ts to return keys as
// labels, so assertions below check for the raw I18n key strings.

const PROFILES: LlmProfileSummary[] = [
  {
    name: "default",
    model: "anthropic/claude-sonnet-4-6",
    base_url: null,
    api_key_set: true,
  },
  {
    name: "gpt-5",
    model: "openai/gpt-5",
    base_url: null,
    api_key_set: true,
  },
];

const renderMenu = (
  overrides: Partial<
    React.ComponentProps<typeof SwitchProfileContextMenu>
  > = {},
) => {
  const profiles = overrides.profiles ?? PROFILES;
  const activeProfileName =
    overrides.activeProfileName !== undefined
      ? overrides.activeProfileName
      : "default";
  const onSelect = overrides.onSelect ?? vi.fn();
  const onClose = overrides.onClose ?? vi.fn();
  render(
    <MemoryRouter>
      <SwitchProfileContextMenu
        profiles={profiles}
        activeProfileName={activeProfileName}
        onSelect={onSelect}
        onClose={onClose}
      />
    </MemoryRouter>,
  );
  return { profiles, activeProfileName, onSelect, onClose };
};

describe("SwitchProfileContextMenu", () => {
  it("renders the i18n header and one row per profile", () => {
    renderMenu();
    expect(
      screen.getByText("SETTINGS$AVAILABLE_PROFILES"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("switch-profile-option-default"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("switch-profile-option-gpt-5"),
    ).toBeInTheDocument();
  });

  it("marks the active profile with aria-current=true and others without", () => {
    renderMenu({ activeProfileName: "gpt-5" });
    expect(screen.getByTestId("switch-profile-option-gpt-5")).toHaveAttribute(
      "aria-current",
      "true",
    );
    expect(
      screen.getByTestId("switch-profile-option-default"),
    ).not.toHaveAttribute("aria-current");
  });

  it("calls onSelect with the profile name and then onClose when a row is clicked", async () => {
    const user = userEvent.setup();
    const props = renderMenu();
    await user.click(screen.getByTestId("switch-profile-option-gpt-5"));
    expect(props.onSelect).toHaveBeenCalledWith("gpt-5");
    expect(props.onClose).toHaveBeenCalled();
  });

  it("renders the Settings entry as a Link to /settings (cmd-click friendly)", () => {
    renderMenu();
    const settings = screen.getByTestId("switch-profile-open-settings");
    expect(settings.tagName).toBe("A");
    expect(settings).toHaveAttribute("href", "/settings");
    expect(settings).toHaveTextContent("MODEL$OPEN_SETTINGS");
  });

  it("calls onClose when clicking the Settings link", async () => {
    const user = userEvent.setup();
    const props = renderMenu();
    await user.click(screen.getByTestId("switch-profile-open-settings"));
    expect(props.onClose).toHaveBeenCalled();
  });

  it("closes the menu when Escape is pressed", async () => {
    const user = userEvent.setup();
    const props = renderMenu();
    await user.keyboard("{Escape}");
    expect(props.onClose).toHaveBeenCalled();
  });
});
