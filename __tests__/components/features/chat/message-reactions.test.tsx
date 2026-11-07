import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import {
  MessageReactions,
  reactionsKey,
} from "#/components/features/chat/message-reactions";
import { featuresService } from "#/api/features-service/features-service.api";
import { useEventStore } from "#/stores/use-event-store";
import type { OHEvent } from "#/stores/use-event-store";

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: "local-1", kind: "local" },
    orgId: null,
  }),
}));

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

const featureState = (enabled: boolean) => [
  {
    key: "conversation_reactions",
    label: "Conversation reactions",
    description: "",
    default: false,
    enabled,
  },
];

describe("reactionsKey", () => {
  it("keeps the latest event per emoji and source, for one message", () => {
    const events = [
      reaction("a", "eyes", "agent"),
      reaction("b", "+1", "user"),
      reaction("c", "+1", "user", true),
      reaction("d", "heart", "user"),
      reaction("e", "eyes", "user"),
      reaction("f", "tada", "agent", false, "other-message"),
    ];

    expect(reactionsKey(events, "msg-1")).toBe(
      "eyes:agent,eyes:user,heart:user",
    );
  });
});

describe("MessageReactions", () => {
  beforeEach(() => {
    useEventStore.getState().clearEvents();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows the agent's reaction on the person's message", async () => {
    vi.spyOn(featuresService, "listFeatures").mockResolvedValue(
      featureState(true),
    );
    useEventStore.getState().addEvents([reaction("a", "eyes", "agent")]);

    renderWithProviders(<MessageReactions eventId="msg-1" author="user" />);

    const chip = await screen.findByTestId("reaction-eyes");
    expect(chip).toHaveAttribute("data-sources", "agent");
    expect(chip).toHaveTextContent("👀");
    // The person reacts to the agent's messages, not to their own.
    expect(screen.queryByTestId("add-reaction")).not.toBeInTheDocument();
  });

  it("renders nothing without reactions while the feature is off", async () => {
    const list = vi
      .spyOn(featuresService, "listFeatures")
      .mockResolvedValue(featureState(false));

    const { container } = renderWithProviders(
      <MessageReactions eventId="msg-1" author="agent" />,
    );

    await waitFor(() => expect(list).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
  });

  it("lets the person react to the agent's message while the feature is on", async () => {
    vi.spyOn(featuresService, "listFeatures").mockResolvedValue(
      featureState(true),
    );
    const react = vi.spyOn(featuresService, "react").mockResolvedValue();
    const user = userEvent.setup();

    renderWithProviders(<MessageReactions eventId="msg-1" author="agent" />);

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
    vi.spyOn(featuresService, "listFeatures").mockResolvedValue(
      featureState(true),
    );
    const react = vi.spyOn(featuresService, "react").mockResolvedValue();
    useEventStore.getState().addEvents([reaction("a", "heart", "user")]);
    const user = userEvent.setup();

    renderWithProviders(<MessageReactions eventId="msg-1" author="agent" />);

    await user.click(await screen.findByTestId("reaction-heart"));

    expect(react).toHaveBeenCalledWith(
      expect.objectContaining({ emoji: "heart", removed: true }),
    );
  });
});
