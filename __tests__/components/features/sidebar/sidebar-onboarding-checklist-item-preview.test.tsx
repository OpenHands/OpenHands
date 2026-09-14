import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { SidebarOnboardingChecklistItemIcon } from "#/components/features/sidebar/sidebar-onboarding-checklist-item-icon";
import { SidebarOnboardingChecklistItemPreview } from "#/components/features/sidebar/sidebar-onboarding-checklist-item-preview";
import {
  NavigationProvider,
  type NavigationContextValue,
} from "#/context/navigation-context";
import { I18nKey } from "#/i18n/declaration";

const navigation: NavigationContextValue = {
  currentPath: "/",
  conversationId: null,
  isNavigating: false,
  navigate: () => undefined,
};

function renderPreview(id: Parameters<typeof SidebarOnboardingChecklistItemPreview>[0]["id"]) {
  return render(
    <NavigationProvider value={navigation}>
      <SidebarOnboardingChecklistItemPreview id={id} />
    </NavigationProvider>,
  );
}

describe("SidebarOnboardingChecklistItemIcon", () => {
  it.each([
    ["configure-llm", "sidebar-onboarding-checklist-icon-configure-llm"],
    ["start-conversation", "sidebar-onboarding-checklist-icon-start-conversation"],
    ["schedule-task", "sidebar-onboarding-checklist-icon-schedule-task"],
    ["customize-agent", "sidebar-onboarding-checklist-icon-customize-agent"],
    ["connect-mcp", "sidebar-onboarding-checklist-icon-connect-mcp"],
    ["join-slack", "sidebar-onboarding-checklist-icon-join-slack"],
  ] as const)("renders an icon for %s", (id, testId) => {
    render(<SidebarOnboardingChecklistItemIcon id={id} />);

    expect(screen.getByTestId(testId)).toBeInTheDocument();
  });
});

describe("SidebarOnboardingChecklistItemPreview", () => {
  it("renders title, icon, action button, and docs link", () => {
    renderPreview("configure-llm");

    expect(
      screen.getByTestId("sidebar-onboarding-checklist-preview-configure-llm"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_CONFIGURE_LLM),
    ).toBeInTheDocument();
    expect(
      screen.getByText(I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_CONFIGURE_LLM_DESC),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("sidebar-onboarding-checklist-preview-action-configure-llm"),
    ).toHaveAttribute("href", "/settings/llm");
    expect(
      screen.getByText(I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_ACTION_CONFIGURE_LLM),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("sidebar-onboarding-checklist-preview-docs-configure-llm"),
    ).toHaveAttribute(
      "href",
      "https://docs.openhands.dev/openhands/usage/settings/llm-settings#llm-profiles",
    );
    expect(
      screen.getByText(I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_DOCS_LINK),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("sidebar-onboarding-checklist-icon-configure-llm"),
    ).toBeInTheDocument();
  });

  it("renders the Slack preview as an external invite action", () => {
    renderPreview("join-slack");

    expect(
      screen.getByTestId("sidebar-onboarding-checklist-preview-join-slack"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("sidebar-onboarding-checklist-preview-action-join-slack"),
    ).toHaveAttribute("href", "https://openhands.dev/joinslack");
    expect(
      screen.getByTestId("sidebar-onboarding-checklist-preview-action-join-slack"),
    ).toHaveAttribute("target", "_blank");
    expect(
      screen.getByTestId("sidebar-onboarding-checklist-preview-docs-join-slack"),
    ).toHaveAttribute("href", "https://docs.openhands.dev/overview/community");
  });
});
