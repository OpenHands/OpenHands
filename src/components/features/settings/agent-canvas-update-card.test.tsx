import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  AGENT_CANVAS_RELEASE_NOTES_URL,
  AGENT_CANVAS_UPDATE_COMMANDS,
} from "#/api/agent-canvas-updates";
import { AGENT_CANVAS_CLIENT_VERSION } from "#/api/client-source";
import { AgentCanvasUpdateCard } from "./agent-canvas-update-card";

const { fetchLatestVersionMock, getLockedCloudHostMock } = vi.hoisted(() => ({
  fetchLatestVersionMock: vi.fn(),
  getLockedCloudHostMock: vi.fn(),
}));

vi.mock("#/api/agent-canvas-updates", async (importOriginal) => ({
  ...(await importOriginal<typeof import("#/api/agent-canvas-updates")>()),
  fetchLatestAgentCanvasVersion: fetchLatestVersionMock,
}));

vi.mock("#/api/agent-server-config", async (importOriginal) => ({
  ...(await importOriginal<typeof import("#/api/agent-server-config")>()),
  getLockedCloudHost: getLockedCloudHostMock,
}));

function renderCard() {
  const queryClient = new QueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <AgentCanvasUpdateCard />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  getLockedCloudHostMock.mockReturnValue(null);
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("AgentCanvasUpdateCard", () => {
  it("shows the running version and an up-to-date badge when the registry matches it", async () => {
    fetchLatestVersionMock.mockResolvedValue(AGENT_CANVAS_CLIENT_VERSION);

    renderCard();

    expect(screen.getByText(AGENT_CANVAS_CLIENT_VERSION)).toBeInTheDocument();
    expect(
      await screen.findByTestId("agent-canvas-update-badge"),
    ).toHaveTextContent("SETTINGS$APP_UPDATE_BADGE_UP_TO_DATE");
  });

  it("flags an available update when the registry reports a newer version", async () => {
    fetchLatestVersionMock.mockResolvedValue("999.0.0");

    renderCard();
    expect(
      await screen.findByTestId("agent-canvas-update-badge"),
    ).toHaveTextContent("SETTINGS$APP_UPDATE_BADGE_UPDATE_AVAILABLE");
    fireEvent.click(screen.getByTestId("agent-canvas-update-toggle"));

    expect(screen.getByTestId("agent-canvas-update-status")).toHaveTextContent(
      "SETTINGS$APP_UPDATE_AVAILABLE_MESSAGE",
    );
  });

  it("treats a running version newer than the registry as up to date", async () => {
    fetchLatestVersionMock.mockResolvedValue("0.0.1");

    renderCard();

    expect(
      await screen.findByTestId("agent-canvas-update-badge"),
    ).toHaveTextContent("SETTINGS$APP_UPDATE_BADGE_UP_TO_DATE");
  });

  it("shows no badge and a quiet inline message when the check fails", async () => {
    fetchLatestVersionMock.mockRejectedValue(new Error("offline"));

    renderCard();
    fireEvent.click(screen.getByTestId("agent-canvas-update-toggle"));

    expect(
      await screen.findByText("SETTINGS$APP_UPDATE_CHECK_FAILED"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("agent-canvas-update-badge"),
    ).not.toBeInTheDocument();
  });

  it("keeps update instructions collapsed until the header is toggled", async () => {
    fetchLatestVersionMock.mockResolvedValue(AGENT_CANVAS_CLIENT_VERSION);

    renderCard();
    expect(
      screen.queryByTestId("agent-canvas-update-command-npm"),
    ).not.toBeInTheDocument();
    fireEvent.click(screen.getByTestId("agent-canvas-update-toggle"));

    expect(
      screen.getByTestId("agent-canvas-update-command-npm"),
    ).toHaveTextContent(AGENT_CANVAS_UPDATE_COMMANDS.npm);
    expect(
      screen.getByTestId("agent-canvas-update-command-docker"),
    ).toHaveTextContent(AGENT_CANVAS_UPDATE_COMMANDS.docker);
    expect(
      screen.getByTestId("agent-canvas-update-release-notes"),
    ).toHaveAttribute("href", AGENT_CANVAS_RELEASE_NOTES_URL);
  });

  it("re-checks the registry when Check for updates is clicked", async () => {
    fetchLatestVersionMock.mockResolvedValue(AGENT_CANVAS_CLIENT_VERSION);

    renderCard();
    await screen.findByTestId("agent-canvas-update-badge");
    expect(fetchLatestVersionMock).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByTestId("agent-canvas-update-toggle"));
    fireEvent.click(screen.getByTestId("agent-canvas-update-check-button"));

    await waitFor(() =>
      expect(fetchLatestVersionMock).toHaveBeenCalledTimes(2),
    );
  });

  it("copies the update command to the clipboard", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(window.navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });
    fetchLatestVersionMock.mockResolvedValue(AGENT_CANVAS_CLIENT_VERSION);

    renderCard();
    fireEvent.click(screen.getByTestId("agent-canvas-update-toggle"));
    // First copy button belongs to the npm command row.
    fireEvent.click(screen.getAllByTestId("copy-to-clipboard")[0]);

    expect(writeText).toHaveBeenCalledWith(AGENT_CANVAS_UPDATE_COMMANDS.npm);
  });

  it("renders nothing and skips the check when locked to a cloud deployment", () => {
    getLockedCloudHostMock.mockReturnValue("https://app.openhands.dev");
    fetchLatestVersionMock.mockResolvedValue(AGENT_CANVAS_CLIENT_VERSION);

    renderCard();

    expect(
      screen.queryByTestId("agent-canvas-update-card"),
    ).not.toBeInTheDocument();
    expect(fetchLatestVersionMock).not.toHaveBeenCalled();
  });
});
