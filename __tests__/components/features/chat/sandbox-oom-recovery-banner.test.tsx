import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { renderWithProviders } from "test-utils";
import { I18nKey } from "#/i18n/declaration";
import { SandboxOOMRecoveryBanner } from "#/components/features/chat/sandbox-oom-recovery-banner";
import { useEventStore } from "#/stores/use-event-store";
import { useSandboxOOMStore } from "#/stores/sandbox-oom-store";
import type { OpenHandsEvent } from "#/types/agent-server/core";

const CONVERSATION_ID = "test-conv-oom";

const makeOOMEvent = (
  id: string,
  exitCode = 137,
  text = "Killed",
): OpenHandsEvent =>
  ({
    id,
    timestamp: "2026-01-01T12:00:00Z",
    source: "environment",
    action_id: "act-1",
    observation: {
      kind: "ExecuteBashObservation",
      content: [{ type: "text", text }],
      exit_code: exitCode,
      metadata: {},
    },
    action: {
      command: "make -j32 build",
    },
  }) as unknown as OpenHandsEvent;

describe("SandboxOOMRecoveryBanner", () => {
  beforeEach(() => {
    useSandboxOOMStore.getState().clearAll();
    useEventStore.getState().clearEvents();
  });

  afterEach(() => {
    useSandboxOOMStore.getState().clearAll();
    useEventStore.getState().clearEvents();
  });

  it("renders nothing when there are no OOM events", () => {
    useEventStore.setState({
      loadedConversationId: CONVERSATION_ID,
      events: [],
    });

    renderWithProviders(
      <SandboxOOMRecoveryBanner conversationId={CONVERSATION_ID} />,
    );
    expect(screen.queryByTestId("sandbox-oom-recovery-banner")).toBeNull();
  });

  it("renders the OOM recovery banner when an OOM event is detected", () => {
    useEventStore.setState({
      loadedConversationId: CONVERSATION_ID,
      events: [makeOOMEvent("oom-1")],
    });

    renderWithProviders(
      <SandboxOOMRecoveryBanner conversationId={CONVERSATION_ID} />,
    );

    const banner = screen.getByTestId("sandbox-oom-recovery-banner");
    expect(banner).toBeInTheDocument();
    expect(
      screen.getByText(I18nKey.CHAT_INTERFACE$SANDBOX_OOM_TITLE),
    ).toBeInTheDocument();
    expect(
      screen.getByText(I18nKey.CHAT_INTERFACE$SANDBOX_OOM_DESCRIPTION),
    ).toBeInTheDocument();
    expect(
      screen.getByText(I18nKey.CHAT_INTERFACE$SANDBOX_OOM_ADVICE_CHUNKS),
    ).toBeInTheDocument();
    expect(
      screen.getByText(I18nKey.CHAT_INTERFACE$SANDBOX_OOM_ADVICE_MEMORY),
    ).toBeInTheDocument();
  });

  it("dismisses the banner when the X button is clicked", async () => {
    const user = userEvent.setup();
    useEventStore.setState({
      loadedConversationId: CONVERSATION_ID,
      events: [makeOOMEvent("oom-2")],
    });

    renderWithProviders(
      <SandboxOOMRecoveryBanner conversationId={CONVERSATION_ID} />,
    );
    expect(
      screen.getByTestId("sandbox-oom-recovery-banner"),
    ).toBeInTheDocument();

    const dismissBtn = screen.getByTestId("sandbox-oom-dismiss-btn");
    await user.click(dismissBtn);

    await waitFor(() => {
      expect(screen.queryByTestId("sandbox-oom-recovery-banner")).toBeNull();
    });

    expect(useSandboxOOMStore.getState().dismissedEventIds["oom-2"]).toBe(true);
  });

  it("dismisses the banner when the bottom Dismiss button is clicked", async () => {
    const user = userEvent.setup();
    useEventStore.setState({
      loadedConversationId: CONVERSATION_ID,
      events: [makeOOMEvent("oom-3")],
    });

    renderWithProviders(
      <SandboxOOMRecoveryBanner conversationId={CONVERSATION_ID} />,
    );
    const bottomDismissBtn = screen.getByTestId(
      "sandbox-oom-bottom-dismiss-btn",
    );
    await user.click(bottomDismissBtn);

    await waitFor(() => {
      expect(screen.queryByTestId("sandbox-oom-recovery-banner")).toBeNull();
    });
  });

  it("renders nothing if conversationId does not match loadedConversationId", () => {
    useEventStore.setState({
      loadedConversationId: "other-conversation",
      events: [makeOOMEvent("oom-4")],
    });

    renderWithProviders(
      <SandboxOOMRecoveryBanner conversationId={CONVERSATION_ID} />,
    );
    expect(screen.queryByTestId("sandbox-oom-recovery-banner")).toBeNull();
  });
});
