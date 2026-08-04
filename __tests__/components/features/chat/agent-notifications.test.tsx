import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AgentNotifications } from "#/components/features/chat/agent-notifications";
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
    createdAt: "2026-01-01T00:00:00.000Z",
  },
];

describe("AgentNotifications", () => {
  const onCreateAll = vi.fn();
  const onDismiss = vi.fn();
  const onRemove = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders agentNotifications with checkboxes checked by default", () => {
    renderWithProviders(
      <AgentNotifications
        agentNotifications={agentNotifications}
        onCreateAll={onCreateAll}
        onDismiss={onDismiss}
        onRemove={onRemove}
      />,
    );

    expect(
      screen.getByTestId("agent-notifications"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("agent-notification-checkbox-skill-standup"),
    ).toBeChecked();
    expect(
      screen.getByTestId(
        "agent-notification-checkbox-workflow-ci",
      ),
    ).toBeChecked();
    expect(
      screen.getByText(I18nKey.CHAT_INTERFACE$AGENT_NOTIFICATIONS_TITLE),
    ).toBeInTheDocument();
    expect(screen.getByTestId("agent-notifications-info")).toBeInTheDocument();
    expect(
      screen.queryByText(I18nKey.CHAT_INTERFACE$AGENT_NOTIFICATIONS_DESCRIPTION),
    ).not.toBeInTheDocument();
  });

  it("calls onCreateAll with only checked agentNotification ids", async () => {
    const user = userEvent.setup();

    renderWithProviders(
      <AgentNotifications
        agentNotifications={agentNotifications}
        onCreateAll={onCreateAll}
        onDismiss={onDismiss}
        onRemove={onRemove}
      />,
    );

    await user.click(
      screen.getByTestId(
        "agent-notification-checkbox-workflow-ci",
      ),
    );
    await user.click(
      screen.getByTestId("agent-notifications-create-all"),
    );

    expect(onCreateAll).toHaveBeenCalledWith(["skill-standup"]);
  });

  it("disables create all when every agentNotification is unchecked", async () => {
    const user = userEvent.setup();

    renderWithProviders(
      <AgentNotifications
        agentNotifications={agentNotifications}
        onCreateAll={onCreateAll}
        onDismiss={onDismiss}
        onRemove={onRemove}
      />,
    );

    await user.click(
      screen.getByTestId("agent-notification-checkbox-skill-standup"),
    );
    await user.click(
      screen.getByTestId(
        "agent-notification-checkbox-workflow-ci",
      ),
    );

    expect(
      screen.getByTestId("agent-notifications-create-all"),
    ).toBeDisabled();
  });

  it("renders all agent notification kinds without crashing", async () => {
    const user = userEvent.setup();
    const allKinds: AgentNotification[] = [
      {
        id: "skill",
        kind: "skill",
        name: "Skill helper",
        prompt: "Save skill.",
        createdAt: "2026-01-01T00:00:00.000Z",
      },
      {
        id: "workflow",
        kind: "workflow",
        name: "Workflow helper",
        prompt: "Create workflow.",
        createdAt: "2026-01-01T00:00:01.000Z",
      },
      {
        id: "routine",
        kind: "routine",
        name: "Routine helper",
        prompt: "Create routine.",
        createdAt: "2026-01-01T00:00:02.000Z",
      },
      {
        id: "responder",
        kind: "responder",
        name: "Responder helper",
        prompt: "Create responder.",
        createdAt: "2026-01-01T00:00:03.000Z",
      },
    ];

    renderWithProviders(
      <AgentNotifications
        agentNotifications={allKinds}
        onCreateAll={onCreateAll}
        onDismiss={onDismiss}
        onRemove={onRemove}
      />,
    );

    expect(screen.getByText("Workflow helper")).toBeInTheDocument();
    for (const id of ["skill", "workflow", "routine", "responder"] as const) {
      expect(
        screen.getByTestId(`agent-notification-kind-pill-${id}`),
      ).toBeInTheDocument();
    }

    await user.click(screen.getByTestId("agent-notification-expand-workflow"));
    expect(
      screen.getByTestId("agent-notification-details-workflow"),
    ).toBeInTheDocument();
    expect(screen.getByText("Create workflow.")).toBeInTheDocument();
  });

  it("expands a row to reveal prompt details", async () => {
    const user = userEvent.setup();

    renderWithProviders(
      <AgentNotifications
        agentNotifications={agentNotifications}
        onCreateAll={onCreateAll}
        onDismiss={onDismiss}
        onRemove={onRemove}
      />,
    );

    expect(
      screen.queryByTestId("agent-notification-details-skill-standup"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("agent-notification-kind-pill-skill-standup"),
    ).toBeInTheDocument();

    await user.click(
      screen.getByTestId("agent-notification-expand-skill-standup"),
    );

    expect(
      screen.getByTestId("agent-notification-details-skill-standup"),
    ).toBeInTheDocument();
    expect(
      screen.getByText('Save a reusable skill named "Standup digest helper".'),
    ).toBeInTheDocument();
  });

  it("calls onDismiss when the footer dismiss button is clicked", async () => {
    const user = userEvent.setup();

    renderWithProviders(
      <AgentNotifications
        agentNotifications={agentNotifications}
        onCreateAll={onCreateAll}
        onDismiss={onDismiss}
        onRemove={onRemove}
      />,
    );

    await user.click(
      screen.getByTestId("agent-notifications-dismiss-action"),
    );

    expect(onDismiss).toHaveBeenCalledTimes(1);
    expect(onCreateAll).not.toHaveBeenCalled();
  });

  it("calls onDismiss when the dismiss button is clicked", async () => {
    const user = userEvent.setup();

    renderWithProviders(
      <AgentNotifications
        agentNotifications={agentNotifications}
        onCreateAll={onCreateAll}
        onDismiss={onDismiss}
        onRemove={onRemove}
      />,
    );

    await user.click(
      screen.getByTestId("agent-notifications-dismiss"),
    );

    expect(onDismiss).toHaveBeenCalledTimes(1);
    expect(onCreateAll).not.toHaveBeenCalled();
  });

  it("calls onRemove when the row remove button is clicked", async () => {
    const user = userEvent.setup();

    renderWithProviders(
      <AgentNotifications
        agentNotifications={agentNotifications}
        onCreateAll={onCreateAll}
        onDismiss={onDismiss}
        onRemove={onRemove}
      />,
    );

    await user.click(
      screen.getByTestId("agent-notification-remove-skill-standup"),
    );

    expect(onRemove).toHaveBeenCalledTimes(1);
    expect(onRemove).toHaveBeenCalledWith("skill-standup");
  });
});
