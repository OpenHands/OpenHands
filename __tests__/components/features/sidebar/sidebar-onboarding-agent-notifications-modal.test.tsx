import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SidebarOnboardingAgentNotificationsModal } from "#/components/features/sidebar/sidebar-onboarding-agent-notifications-modal";
import type { AgentNotification } from "#/components/features/chat/agent-notifications.constants";
import { I18nKey } from "#/i18n/declaration";
import { renderWithProviders } from "test-utils";

const agentNotifications: AgentNotification[] = [
  {
    id: "skill-standup",
    kind: "skill",
    name: "Standup digest helper",
    prompt: 'Save a reusable skill named "Standup digest helper".',
    createdAt: "2026-01-01T00:00:00.000Z",
  },
  {
    id: "workflow-ci",
    kind: "workflow",
    name: "CI failure watchdog",
    prompt: "Create a workflow named CI failure watchdog.",
    createdAt: "2026-01-01T00:00:01.000Z",
  },
];

describe("SidebarOnboardingAgentNotificationsModal", () => {
  const onCreateAll = vi.fn();
  const onClose = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders agentNotifications with checkboxes checked by default", () => {
    renderWithProviders(
      <SidebarOnboardingAgentNotificationsModal
        agentNotifications={agentNotifications}
        isOpen
        onClose={onClose}
        onCreateAll={onCreateAll}
      />,
    );

    expect(
      screen.getByTestId("sidebar-onboarding-agent-notifications-modal"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId(
        "sidebar-onboarding-agent-notification-checkbox-skill-standup",
      ),
    ).toBeChecked();
    expect(
      screen.getByText(I18nKey.CHAT_INTERFACE$AGENT_NOTIFICATIONS_TITLE),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("sidebar-onboarding-agent-notifications-info"),
    ).toBeInTheDocument();
    expect(
      screen.queryByText(I18nKey.CHAT_INTERFACE$AGENT_NOTIFICATIONS_DESCRIPTION),
    ).not.toBeInTheDocument();
  });

  it("calls onCreateAll with only checked agentNotification ids and closes", async () => {
    const user = userEvent.setup();

    renderWithProviders(
      <SidebarOnboardingAgentNotificationsModal
        agentNotifications={agentNotifications}
        isOpen
        onClose={onClose}
        onCreateAll={onCreateAll}
      />,
    );

    await user.click(
      screen.getByTestId(
        "sidebar-onboarding-agent-notification-checkbox-workflow-ci",
      ),
    );
    await user.click(
      screen.getByTestId("sidebar-onboarding-agent-notifications-create-all"),
    );

    expect(onCreateAll).toHaveBeenCalledWith(["skill-standup"]);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("does not render when closed", () => {
    renderWithProviders(
      <SidebarOnboardingAgentNotificationsModal
        agentNotifications={agentNotifications}
        isOpen={false}
        onClose={onClose}
        onCreateAll={onCreateAll}
      />,
    );

    expect(
      screen.queryByTestId("sidebar-onboarding-agent-notifications-modal"),
    ).not.toBeInTheDocument();
  });
});
