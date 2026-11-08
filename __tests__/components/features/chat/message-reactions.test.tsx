import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { HttpError } from "@openhands/typescript-client";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import {
  MessageReactions,
  useMessageReactions,
} from "#/components/features/chat/message-reactions";
import { experimentalFeaturesService } from "#/api/experimental-features-service/experimental-features-service.api";
import { useEventStore } from "#/stores/use-event-store";
import type { OHEvent } from "#/stores/use-event-store";

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: "local-1", kind: "local" },
    orgId: null,
  }),
}));

const { displayApiErrorToast } = vi.hoisted(() => ({
  displayApiErrorToast: vi.fn(),
}));
vi.mock("#/utils/custom-toast-handlers", () => ({ displayApiErrorToast }));

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: "conv-1" }),
}));

const reaction = (
  id: string,
  emoji: string,
  source: "agent" | "user",
  removed = false,
  target = "msg-1",
) =>
  ({
    id,
    kind: "ReactionEvent",
    timestamp: `2026-11-07T00:00:0${id.length}Z`,
    source,
    target_event_id: target,
    emoji,
    removed,
  }) as unknown as OHEvent;

function Row({
  eventId = "msg-1",
  author,
  isFromPlanningAgent = false,
}: {
  eventId?: string;
  author: "user" | "agent";
  isFromPlanningAgent?: boolean;
}) {
  const state = useMessageReactions(eventId, author, { isFromPlanningAgent });
  return (
    <div data-testid="row" data-visible={state.visible}>
      {state.visible && <MessageReactions state={state} />}
    </div>
  );
}

const featureState = (enabled: boolean) => [
  {
    key: "conversation_reactions",
    label: "Conversation reactions",
    description: "",
    default: false,
    enabled,
  },
];

describe("MessageReactions", () => {
  beforeEach(() => {
    useEventStore.getState().clearEvents();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows the agent's reaction on the person's message", async () => {
    vi.spyOn(experimentalFeaturesService, "listExperimentalFeatures").mockResolvedValue(
      featureState(true),
    );
    useEventStore.getState().addEvents([reaction("a", "eyes", "agent")]);

    renderWithProviders(<Row author="user" />);

    const chip = await screen.findByTestId("reaction-eyes");
    expect(chip).toHaveAttribute("data-sources", "agent");
    expect(chip).toHaveTextContent("👀");
    // The person reacts to the agent's messages, not to their own.
    expect(screen.queryByTestId("add-reaction")).not.toBeInTheDocument();
  });

  it("keeps the latest reaction per emoji and source, for one message", async () => {
    vi.spyOn(experimentalFeaturesService, "listExperimentalFeatures").mockResolvedValue(
      featureState(false),
    );
    useEventStore
      .getState()
      .addEvents([
        reaction("a", "eyes", "agent"),
        reaction("b", "+1", "user"),
        reaction("cc", "+1", "user", true),
        reaction("ddd", "heart", "user"),
        reaction("eeee", "eyes", "user"),
        reaction("fffff", "tada", "agent", false, "other-message"),
      ]);

    renderWithProviders(<Row author="agent" />);

    expect(await screen.findByTestId("reaction-eyes")).toHaveAttribute(
      "data-sources",
      "agent,user",
    );
    expect(screen.getByTestId("reaction-heart")).toBeInTheDocument();
    expect(screen.queryByTestId("reaction-+1")).not.toBeInTheDocument();
    expect(screen.queryByTestId("reaction-tada")).not.toBeInTheDocument();
  });

  it("shows no row on the agent's message while the feature is off, and a picker once it is on", async () => {
    const list = vi
      .spyOn(experimentalFeaturesService, "listExperimentalFeatures")
      .mockResolvedValue(featureState(false));

    const off = renderWithProviders(<Row author="agent" />);
    await waitFor(() => expect(list).toHaveBeenCalled());
    await waitFor(() =>
      expect(screen.getByTestId("row")).toHaveAttribute("data-visible", "false"),
    );
    expect(screen.queryByTestId("add-reaction")).not.toBeInTheDocument();
    off.unmount();

    list.mockResolvedValue(featureState(true));
    renderWithProviders(<Row author="agent" />);
    expect(await screen.findByTestId("add-reaction")).toBeInTheDocument();
  });

  it("does not offer reactions on a planning agent's message", async () => {
    const list = vi
      .spyOn(experimentalFeaturesService, "listExperimentalFeatures")
      .mockResolvedValue(featureState(true));

    renderWithProviders(<Row author="agent" isFromPlanningAgent />);

    await waitFor(() => expect(list).toHaveBeenCalled());
    expect(screen.getByTestId("row")).toHaveAttribute("data-visible", "false");
  });

  it("closes the picker on Escape and returns focus to its button", async () => {
    vi.spyOn(experimentalFeaturesService, "listExperimentalFeatures").mockResolvedValue(
      featureState(true),
    );
    const user = userEvent.setup();

    renderWithProviders(<Row author="agent" />);

    const trigger = await screen.findByTestId("add-reaction");
    await user.click(trigger);
    expect(screen.getByTestId("reaction-picker-eyes")).toHaveFocus();
    await user.keyboard("{Escape}");
    expect(screen.queryByTestId("reaction-picker")).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });

  it("shows the server's reason when a reaction fails", async () => {
    vi.spyOn(experimentalFeaturesService, "listExperimentalFeatures").mockResolvedValue(
      featureState(true),
    );
    vi.spyOn(experimentalFeaturesService, "react").mockRejectedValue(
      new HttpError(409, "Conflict", {
        detail: "Conversation reactions are turned off",
      }),
    );
    const user = userEvent.setup();

    renderWithProviders(<Row author="agent" />);

    await user.click(await screen.findByTestId("add-reaction"));
    await user.click(screen.getByTestId("reaction-picker-heart"));

    // The helper reads the server's `detail`, here "Conversation reactions
    // are turned off", and falls back to the translated message.
    await waitFor(() =>
      expect(displayApiErrorToast).toHaveBeenCalledWith(
        expect.objectContaining({ status: 409 }),
        "CHAT_INTERFACE$REACTION_FAILED",
      ),
    );
  });

  it("lets the person react to the agent's message while the feature is on", async () => {
    vi.spyOn(experimentalFeaturesService, "listExperimentalFeatures").mockResolvedValue(
      featureState(true),
    );
    const react = vi.spyOn(experimentalFeaturesService, "react").mockResolvedValue();
    const user = userEvent.setup();

    renderWithProviders(<Row author="agent" />);

    await user.click(await screen.findByTestId("add-reaction"));
    await user.click(screen.getByTestId("reaction-picker-+1"));

    expect(react).toHaveBeenCalledWith({
      conversationId: "conv-1",
      emoji: "+1",
      targetEventId: "msg-1",
      removed: false,
    });
  });

  it("takes the person's own reaction off when its chip is clicked", async () => {
    vi.spyOn(experimentalFeaturesService, "listExperimentalFeatures").mockResolvedValue(
      featureState(true),
    );
    const react = vi.spyOn(experimentalFeaturesService, "react").mockResolvedValue();
    useEventStore.getState().addEvents([reaction("a", "heart", "user")]);
    const user = userEvent.setup();

    renderWithProviders(<Row author="agent" />);

    await user.click(await screen.findByTestId("reaction-heart"));

    expect(react).toHaveBeenCalledWith(
      expect.objectContaining({ emoji: "heart", removed: true }),
    );
  });
});
