import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { SidebarOnboardingChecklistItemIcon } from "#/components/features/sidebar/sidebar-onboarding-checklist-item-icon";
import { SidebarOnboardingChecklistItemPreview } from "#/components/features/sidebar/sidebar-onboarding-checklist-item-preview";
import {
  NavigationProvider,
  type NavigationContextValue,
} from "#/context/navigation-context";
import { I18nKey } from "#/i18n/declaration";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { isConfigureLlmChecklistItemComplete } from "#/components/features/sidebar/sidebar-onboarding-checklist-llm-complete";

describe("sidebar-onboarding-checklist-item-preview", () => {
  const navigation: NavigationContextValue = {
    currentPath: "/",
    conversationId: null,
    isNavigating: false,
    navigate: () => undefined,
  };

  function renderPreview(
    id: Parameters<typeof SidebarOnboardingChecklistItemPreview>[0]["id"],
  ) {
    return render(
      <NavigationProvider value={navigation}>
        <SidebarOnboardingChecklistItemPreview id={id} />
      </NavigationProvider>,
    );
  }

  describe("SidebarOnboardingChecklistItemIcon", () => {
    it.each([
      ["configure-llm", "sidebar-onboarding-checklist-icon-configure-llm"],
      [
        "start-conversation",
        "sidebar-onboarding-checklist-icon-start-conversation",
      ],
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
        screen.getByTestId(
          "sidebar-onboarding-checklist-preview-configure-llm",
        ),
      ).toBeInTheDocument();
      expect(
        screen.getByText(I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_CONFIGURE_LLM),
      ).toBeInTheDocument();
      expect(
        screen.getByText(
          I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_CONFIGURE_LLM_DESC,
        ),
      ).toBeInTheDocument();
      expect(
        screen.getByTestId(
          "sidebar-onboarding-checklist-preview-action-configure-llm",
        ),
      ).toHaveAttribute("href", "/settings/llm");
      expect(
        screen.getByText(
          I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_ACTION_CONFIGURE_LLM,
        ),
      ).toBeInTheDocument();
      expect(
        screen.getByTestId(
          "sidebar-onboarding-checklist-preview-docs-configure-llm",
        ),
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
        screen.getByTestId(
          "sidebar-onboarding-checklist-preview-action-join-slack",
        ),
      ).toHaveAttribute("href", "https://openhands.dev/joinslack");
      expect(
        screen.getByTestId(
          "sidebar-onboarding-checklist-preview-action-join-slack",
        ),
      ).toHaveAttribute("target", "_blank");
      expect(
        screen.getByTestId(
          "sidebar-onboarding-checklist-preview-docs-join-slack",
        ),
      ).toHaveAttribute(
        "href",
        "https://docs.openhands.dev/overview/community",
      );
    });
  });
});

describe("isConfigureLlmChecklistItemComplete", () => {
  it("returns false while LLM readiness is still indeterminate", () => {
    expect(
      isConfigureLlmChecklistItemComplete(
        DEFAULT_SETTINGS,
        false,
        true,
        undefined,
        true,
      ),
    ).toBe(false);
  });

  it("returns true when useLlmConfigured reports configured", () => {
    expect(
      isConfigureLlmChecklistItemComplete(
        DEFAULT_SETTINGS,
        true,
        false,
        undefined,
        false,
      ),
    ).toBe(true);
  });

  it("returns true when any saved LLM profile has an API key", () => {
    expect(
      isConfigureLlmChecklistItemComplete(
        DEFAULT_SETTINGS,
        false,
        true,
        {
          active_profile: "work",
          profiles: [
            {
              name: "work",
              model: "openai/gpt-5.5",
              base_url: "https://api.openai.com/v1",
              api_key_set: true,
            },
          ],
        },
        false,
      ),
    ).toBe(true);
  });

  it("returns true when settings already have a model and API key", () => {
    expect(
      isConfigureLlmChecklistItemComplete(
        {
          ...DEFAULT_SETTINGS,
          llm_api_key_set: true,
          agent_settings: {
            ...(DEFAULT_SETTINGS.agent_settings ?? {}),
            llm: { model: "openai/gpt-5.5" },
          },
        },
        false,
        true,
        undefined,
        true,
      ),
    ).toBe(true);
  });

  it("returns false when no model or auth is present", () => {
    expect(
      isConfigureLlmChecklistItemComplete(
        DEFAULT_SETTINGS,
        false,
        false,
        { active_profile: null, profiles: [] },
        false,
      ),
    ).toBe(false);
  });
});
