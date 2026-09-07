import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import ChannelService from "#/api/channel-service/channel-service.api";
import type {
  ChannelMessagePage,
  ChannelRecord,
} from "#/api/channel-service/channel-types";
import { ChannelsOverview } from "#/components/features/channels/channels-overview";
import { ChannelsSubpageLayout } from "#/components/features/channels/channels-subpage-layout";
import { I18nKey } from "#/i18n/declaration";

const SLACK: ChannelRecord = {
  id: "slack",
  type: "slack",
  status: { state: "stopped", mode: "sockets", metrics: { posted: 0 } },
  config: { routing_rules: [], human_only: false, cost_cap: null },
};

const EMPTY_PAGE: ChannelMessagePage = { items: [], limit: 50, offset: 0 };

describe("ChannelsOverview", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(ChannelService, "list").mockResolvedValue([SLACK]);
    vi.spyOn(ChannelService, "listMessages").mockResolvedValue(EMPTY_PAGE);
  });

  it("starts a stopped channel", async () => {
    const user = userEvent.setup();
    const start = vi.spyOn(ChannelService, "start").mockResolvedValue({
      ...SLACK,
      status: { ...SLACK.status, state: "running" },
    });

    renderWithProviders(<ChannelsOverview />);

    expect(await screen.findByTestId("channel-row-slack")).toBeInTheDocument();
    await user.click(screen.getByTestId("channel-toggle-slack"));
    await waitFor(() => expect(start).toHaveBeenCalledWith("slack"));
  });

  it("opens config and saves routing rules", async () => {
    const user = userEvent.setup();
    const updateConfig = vi
      .spyOn(ChannelService, "updateConfig")
      .mockResolvedValue(SLACK);

    renderWithProviders(<ChannelsOverview />);
    await screen.findByTestId("channel-row-slack");
    await user.click(screen.getByTestId("channel-config-slack"));
    await user.click(screen.getByTestId("channel-config-save-slack"));
    await waitFor(() =>
      expect(updateConfig).toHaveBeenCalledWith("slack", expect.any(Object)),
    );
  });
});

describe("channel i18n keys", () => {
  it("exposes the overview copy keys", () => {
    expect(I18nKey.CHANNELS$NAV).toBe("CHANNELS$NAV");
    expect(I18nKey.CHANNELS$CONNECTIONS).toBe("CHANNELS$CONNECTIONS");
    expect(I18nKey.CHANNELS$START).toBe("CHANNELS$START");
  });
});

describe("ChannelsSubpageLayout", () => {
  it("mirrors Customize/Automate with an inner aside for channels, messages, and Meetily", () => {
    renderWithProviders(
      <ChannelsSubpageLayout>
        <div />
      </ChannelsSubpageLayout>,
    );

    const nav = screen.getByTestId("channels-navbar-desktop");
    expect(nav).toBeInTheDocument();
    expect(
      within(nav).getByTestId("channels-navigation-channels"),
    ).toHaveAttribute("href", "/channels");
    expect(
      within(nav).getByTestId("channels-navigation-messages"),
    ).toHaveAttribute("href", "/channels/messages");
    expect(
      within(nav).getByTestId("channels-navigation-meetily"),
    ).toHaveAttribute("href", "/channels/meetily");
  });
});
