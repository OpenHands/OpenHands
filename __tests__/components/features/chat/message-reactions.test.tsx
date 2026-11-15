import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, screen, waitFor } from "@testing-library/react";
import {
  HttpError,
  type ReactionEmoji,
  type ReactionEvent,
} from "@openhands/typescript-client";
import userEvent from "@testing-library/user-event";
import {
  ConversationClient,
  SettingsClient,
} from "@openhands/typescript-client/clients";
import { renderWithProviders } from "test-utils";
import {
  MessageReactions,
  useMessageReactions,
} from "#/components/features/chat/message-reactions";
import { experimentalFeaturesService } from "#/api/experimental-features-service/experimental-features-service.api";
import { conversationReactionsService } from "#/api/conversation-reactions-service/conversation-reactions-service.api";
import { useEventStore } from "#/stores/use-event-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";

const { displayApiErrorToast } = vi.hoisted(() => ({
  displayApiErrorToast: vi.fn(),
}));
vi.mock("#/utils/custom-toast-handlers", () => ({ displayApiErrorToast }));

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: "conv-1" }),
}));

const reaction = (
  id: string,
  emoji: ReactionEmoji,
  source: "agent" | "user",
  removed = false,
  target = "msg-1",
): ReactionEvent => ({
  id,
  kind: "ReactionEvent",
  timestamp: `2026-11-07T00:00:0${id.length}Z`,
  source,
  target_event_id: target,
  emoji,
  removed,
});

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
    setRegisteredBackends([
      {
        id: "local",
        name: "Local",
        kind: "local",
        host: "http://local.test",
        apiKey: "",
      },
    ]);
    setActiveSelection({ backendId: "local", orgId: null });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("isolates an enabled Local flag from Cloud and restores it on return", async () => {
    const list = vi
      .spyOn(SettingsClient.prototype, "listExperimentalFeatures")
      .mockResolvedValue(featureState(true));
    setRegisteredBackends([
      {
        id: "local",
        name: "Local",
        kind: "local",
        host: "http://local.test",
        apiKey: "",
      },
      {
        id: "cloud",
        name: "Cloud",
        kind: "cloud",
        host: "https://cloud.test",
        apiKey: "",
      },
    ]);
    setActiveSelection({ backendId: "local", orgId: null });
    renderWithProviders(
      <ActiveBackendProvider>
        <Row author="agent" />
      </ActiveBackendProvider>,
    );
    expect(await screen.findByTestId("add-reaction")).toBeInTheDocument();
    list.mockClear();

    act(() => setActiveSelection({ backendId: "cloud", orgId: "qa-org" }));
    expect(screen.queryByTestId("add-reaction")).not.toBeInTheDocument();
    expect(list).not.toHaveBeenCalled();

    act(() => setActiveSelection({ backendId: "local", orgId: null }));
    expect(await screen.findByTestId("add-reaction")).toBeInTheDocument();
  });

  it("shows the agent's reaction on the user's message", async () => {
    vi.spyOn(
      experimentalFeaturesService,
      "listExperimentalFeatures",
    ).mockResolvedValue(featureState(true));
    useEventStore.getState().addEvents([reaction("a", "eyes", "agent")]);

    renderWithProviders(<Row author="user" />);

    const chip = await screen.findByTestId("reaction-eyes");
    expect(chip).toHaveAttribute("data-sources", "agent");
    expect(chip).toHaveTextContent("👀");
    // The user reacts to the agent's messages, not to their own.
    expect(screen.queryByTestId("add-reaction")).not.toBeInTheDocument();
  });

  it("keeps the latest reaction per emoji and source, for one message", async () => {
    vi.spyOn(
      experimentalFeaturesService,
      "listExperimentalFeatures",
    ).mockResolvedValue(featureState(false));
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
      expect(screen.getByTestId("row")).toHaveAttribute(
        "data-visible",
        "false",
      ),
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
    vi.spyOn(
      experimentalFeaturesService,
      "listExperimentalFeatures",
    ).mockResolvedValue(featureState(true));
    const user = userEvent.setup();

    renderWithProviders(<Row author="agent" />);

    const trigger = await screen.findByTestId("add-reaction");
    await user.click(trigger);
    expect(screen.getByTestId("reaction-picker-eyes")).toHaveFocus();
    await user.keyboard("{Escape}");
    expect(screen.queryByTestId("reaction-picker")).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });

  it("opens the picker above its button, inside the chat, when it would overflow a narrow screen", async () => {
    vi.spyOn(
      experimentalFeaturesService,
      "listExperimentalFeatures",
    ).mockResolvedValue(featureState(true));
    vi.spyOn(window, "innerWidth", "get").mockReturnValue(390);
    const box = (left: number, width: number) =>
      ({
        left,
        right: left + width,
        width,
        top: 0,
        bottom: 24,
        height: 24,
      }) as DOMRect;
    const rect = vi
      .spyOn(HTMLElement.prototype, "getBoundingClientRect")
      .mockImplementation(function measure(this: HTMLElement) {
        // Five chips push the button to x=226; beside it the picker would end at 418.
        if (this.dataset.testid === "reaction-picker") return box(254, 164);
        return box(226, 24);
      });
    const user = userEvent.setup();

    renderWithProviders(<Row author="agent" />);
    await user.click(await screen.findByTestId("add-reaction"));

    const picker = screen.getByTestId("reaction-picker");
    await waitFor(() =>
      expect(picker).toHaveAttribute("data-placement", "above"),
    );
    // Right edge kept 8px inside the 390px window: 390 - 8 - 164 = 218.
    expect(picker.style.getPropertyValue("--picker-shift")).toBe("-8px");
    rect.mockRestore();
  });

  it("opens the picker beside its button when it fits", async () => {
    vi.spyOn(
      experimentalFeaturesService,
      "listExperimentalFeatures",
    ).mockResolvedValue(featureState(true));
    const user = userEvent.setup();

    renderWithProviders(<Row author="agent" />);
    await user.click(await screen.findByTestId("add-reaction"));

    expect(screen.getByTestId("reaction-picker")).toHaveAttribute(
      "data-placement",
      "beside",
    );
  });

  it("shows the server's reason when a reaction fails", async () => {
    vi.spyOn(
      experimentalFeaturesService,
      "listExperimentalFeatures",
    ).mockResolvedValue(featureState(true));
    vi.spyOn(conversationReactionsService, "react").mockRejectedValue(
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
    expect(screen.queryByTestId("reaction-heart")).not.toBeInTheDocument();
  });

  it("lets the user react to the agent's message while the feature is on", async () => {
    vi.spyOn(
      experimentalFeaturesService,
      "listExperimentalFeatures",
    ).mockResolvedValue(featureState(true));
    const react = vi
      .spyOn(conversationReactionsService, "react")
      .mockImplementation(async ({ emoji, removed }) =>
        reaction(removed ? "bb" : "a", emoji, "user", removed),
      );
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
    // The chip shows before the server's event arrives (the server holds a
    // reaction back while a tool call waits for approval), and the event
    // then takes over.
    const chip = await screen.findByTestId("reaction-+1");
    expect(chip).toHaveAttribute("aria-pressed", "true");

    await user.click(chip);
    expect(react).toHaveBeenLastCalledWith(
      expect.objectContaining({ emoji: "+1", removed: true }),
    );
    await waitFor(() =>
      expect(screen.queryByTestId("reaction-+1")).not.toBeInTheDocument(),
    );
  });

  it("takes the user's own reaction off when its chip is clicked", async () => {
    vi.spyOn(
      experimentalFeaturesService,
      "listExperimentalFeatures",
    ).mockResolvedValue(featureState(true));
    const react = vi
      .spyOn(conversationReactionsService, "react")
      .mockResolvedValue(reaction("bb", "heart", "user", true));
    useEventStore.getState().addEvents([reaction("a", "heart", "user")]);
    const user = userEvent.setup();

    renderWithProviders(<Row author="agent" />);

    await user.click(await screen.findByTestId("reaction-heart"));

    expect(react).toHaveBeenCalledWith(
      expect.objectContaining({ emoji: "heart", removed: true }),
    );
  });

  it("keeps other optimistic choices when one request is acknowledged", async () => {
    vi.spyOn(
      experimentalFeaturesService,
      "listExperimentalFeatures",
    ).mockResolvedValue(featureState(true));
    const heartAdded = reaction("a", "heart", "user");
    const thumbAdded = reaction("bb", "+1", "user");
    const heartRemoved = reaction("ccc", "heart", "user", true);
    const react = vi.spyOn(ConversationClient.prototype, "react");
    react.mockResolvedValueOnce(heartAdded);
    let resolveThumb!: (event: ReactionEvent) => void;
    react.mockImplementationOnce(
      () =>
        new Promise<ReactionEvent>((resolve) => {
          resolveThumb = resolve;
        }),
    );
    react.mockResolvedValueOnce(heartRemoved);
    const user = userEvent.setup();
    renderWithProviders(<Row author="agent" />);
    const trigger = await screen.findByTestId("add-reaction");

    await user.click(trigger);
    await user.click(screen.getByTestId("reaction-picker-heart"));
    await user.click(trigger);
    await user.click(screen.getByTestId("reaction-picker-+1"));
    act(() => useEventStore.getState().addEvent(heartAdded));
    expect(screen.getByTestId("reaction-+1")).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    await act(async () => resolveThumb(thumbAdded));
    await user.click(screen.getByTestId("reaction-heart"));
    expect(screen.queryByTestId("reaction-heart")).not.toBeInTheDocument();

    act(() =>
      useEventStore.getState().addEvent(reaction("dddd", "eyes", "agent")),
    );
    expect(screen.getByTestId("reaction-+1")).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.queryByTestId("reaction-heart")).not.toBeInTheDocument();
    act(() => useEventStore.getState().addEvents([thumbAdded, heartRemoved]));
    expect(screen.getByTestId("reaction-+1")).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.queryByTestId("reaction-heart")).not.toBeInTheDocument();
    // Once acknowledged, subsequent server changes must take over.
    act(() =>
      useEventStore.getState().addEvent(reaction("eeeee", "+1", "user", true)),
    );
    expect(screen.queryByTestId("reaction-+1")).not.toBeInTheDocument();
  });

  it("does not resurrect a removed reaction when its earlier add arrives late", async () => {
    vi.spyOn(
      experimentalFeaturesService,
      "listExperimentalFeatures",
    ).mockResolvedValue(featureState(true));
    const added = reaction("a", "heart", "user");
    const removed = reaction("bb", "heart", "user", true);
    vi.spyOn(ConversationClient.prototype, "react")
      .mockResolvedValueOnce(added)
      .mockResolvedValueOnce(removed);
    const user = userEvent.setup();
    renderWithProviders(<Row author="agent" />);

    await user.click(await screen.findByTestId("add-reaction"));
    await user.click(screen.getByTestId("reaction-picker-heart"));
    await user.click(screen.getByTestId("reaction-heart"));
    act(() => useEventStore.getState().addEvent(added));
    expect(screen.queryByTestId("reaction-heart")).not.toBeInTheDocument();
    act(() => useEventStore.getState().addEvent(removed));
    expect(screen.queryByTestId("reaction-heart")).not.toBeInTheDocument();
    act(() =>
      useEventStore.getState().addEvent(reaction("ccc", "heart", "user")),
    );
    expect(screen.getByTestId("reaction-heart")).toHaveAttribute(
      "aria-pressed",
      "true",
    );
  });

  it("rolls back only the failed choice when another acknowledgement arrives during the request", async () => {
    vi.spyOn(
      experimentalFeaturesService,
      "listExperimentalFeatures",
    ).mockResolvedValue(featureState(true));
    const heartAdded = reaction("a", "heart", "user");
    let rejectThumb!: (error: Error) => void;
    vi.spyOn(ConversationClient.prototype, "react")
      .mockResolvedValueOnce(heartAdded)
      .mockImplementationOnce(
        () =>
          new Promise<ReactionEvent>((_, reject) => {
            rejectThumb = reject;
          }),
      );
    const user = userEvent.setup();
    renderWithProviders(<Row author="agent" />);
    const trigger = await screen.findByTestId("add-reaction");
    await user.click(trigger);
    await user.click(screen.getByTestId("reaction-picker-heart"));
    await user.click(trigger);
    await user.click(screen.getByTestId("reaction-picker-+1"));
    act(() => useEventStore.getState().addEvent(heartAdded));
    await act(async () => rejectThumb(new Error("request failed")));
    expect(screen.queryByTestId("reaction-+1")).not.toBeInTheDocument();
    act(() =>
      useEventStore.getState().addEvent(reaction("bb", "heart", "user", true)),
    );
    expect(screen.queryByTestId("reaction-heart")).not.toBeInTheDocument();
  });

  it("keeps keyboard focus at the trigger and prevents activation while saving", async () => {
    vi.spyOn(
      experimentalFeaturesService,
      "listExperimentalFeatures",
    ).mockResolvedValue(featureState(true));
    let resolveReaction!: (event: ReactionEvent) => void;
    const react = vi
      .spyOn(ConversationClient.prototype, "react")
      .mockImplementation(
        () =>
          new Promise<ReactionEvent>((resolve) => {
            resolveReaction = resolve;
          }),
      );
    const user = userEvent.setup();
    renderWithProviders(<Row author="agent" />);
    const trigger = await screen.findByTestId("add-reaction");
    trigger.focus();
    await user.keyboard("{Enter}");
    await user.keyboard("{Enter}");

    expect(trigger).toHaveFocus();
    expect(trigger).not.toBeDisabled();
    expect(trigger).toHaveAttribute("aria-disabled", "true");
    await user.keyboard("{Enter}");
    expect(screen.queryByTestId("reaction-picker")).not.toBeInTheDocument();
    const added = reaction("a", "eyes", "user");
    // The WebSocket can beat the HTTP response as well as follow it.
    act(() => useEventStore.getState().addEvent(added));
    await act(async () => resolveReaction(added));
    expect(trigger).toHaveFocus();
    act(() =>
      useEventStore.getState().addEvent(reaction("bb", "eyes", "user", true)),
    );
    expect(screen.queryByTestId("reaction-eyes")).not.toBeInTheDocument();
    await user.keyboard("{Enter}");
    expect(screen.getByTestId("reaction-picker")).toBeInTheDocument();
    expect(react).toHaveBeenCalledTimes(1);
  });
});
