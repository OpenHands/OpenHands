import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SayHelloStep } from "#/components/features/onboarding/steps/say-hello-step";
import { SIDEBAR_ONBOARDING_CHECKLIST_DISMISSED_STORAGE_KEY } from "#/components/features/sidebar/sidebar-onboarding-checklist.constants";
import { readSidebarOnboardingChecklistDismissed } from "#/components/features/sidebar/sidebar-onboarding-checklist-storage";
import { NavigationProvider } from "#/context/navigation-context";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";

vi.mock("#/hooks/use-is-creating-conversation", () => ({
  useIsCreatingConversation: () => false,
}));

vi.mock("#/hooks/mutation/use-create-conversation", () => ({
  useCreateConversation: () => ({
    mutate: vi.fn(),
    isPending: false,
    isSuccess: false,
  }),
}));

vi.mock(
  "#/components/features/automations/recommended-automations-launcher",
  () => ({
    RecommendedAutomationsLauncher: ({
      onLaunched,
    }: {
      onLaunched?: () => void;
    }) => (
      <button type="button" onClick={onLaunched}>
        launch recommended automation
      </button>
    ),
  }),
);

function renderSayHelloStep(
  props: Partial<React.ComponentProps<typeof SayHelloStep>> = {},
) {
  const onBack = vi.fn();
  const onClose = vi.fn();
  const onLaunched = vi.fn();

  render(
    <ActiveBackendProvider>
      <NavigationProvider
        value={{
          currentPath: "/",
          conversationId: null,
          isNavigating: false,
          navigate: vi.fn(),
        }}
      >
        <SayHelloStep
          onBack={onBack}
          onClose={onClose}
          onLaunched={onLaunched}
          {...props}
        />
      </NavigationProvider>
    </ActiveBackendProvider>,
  );

  return { onBack, onClose, onLaunched };
}

describe("SayHelloStep", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("shows the getting started checklist opt-in checked by default", () => {
    renderSayHelloStep();

    expect(
      screen.getByTestId("onboarding-show-getting-started-checklist"),
    ).toBeChecked();
  });

  it("persists an opt-out when the user closes the final onboarding step", async () => {
    const user = userEvent.setup();
    const { onClose } = renderSayHelloStep();

    await user.click(
      screen.getByTestId("onboarding-show-getting-started-checklist"),
    );
    await user.click(screen.getByTestId("onboarding-hello-close"));

    expect(onClose).toHaveBeenCalledTimes(1);
    expect(readSidebarOnboardingChecklistDismissed()).toBe(true);
    expect(
      window.localStorage.getItem(
        SIDEBAR_ONBOARDING_CHECKLIST_DISMISSED_STORAGE_KEY,
      ),
    ).toBe("true");
  });

  it("keeps the checklist enabled when the user leaves it checked", async () => {
    const user = userEvent.setup();
    renderSayHelloStep();

    await user.click(screen.getByTestId("onboarding-hello-close"));

    expect(readSidebarOnboardingChecklistDismissed()).toBe(false);
  });
});
