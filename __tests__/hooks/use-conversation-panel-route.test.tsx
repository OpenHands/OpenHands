import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { MemoryRouter, Route, Routes, useLocation } from "react-router";
import { useConversationPanelRoute } from "#/hooks/use-conversation-panel-route";
import { handleCanvasUIAction } from "#/services/canvas-ui";
import { useConversationStore } from "#/stores/conversation-store";
import type { CanvasUIAction } from "#/types/agent-server/core";

const CONVERSATION_ID = "conv-1";
const CHAT_PATH = `/conversations/${CONVERSATION_ID}`;
const PANEL_PATH = `${CHAT_PATH}/panel`;
const PHONE_WIDTH = 390;
const DESKTOP_WIDTH = 1440;

function setWindowWidth(width: number) {
  Object.defineProperty(window, "innerWidth", {
    writable: true,
    configurable: true,
    value: width,
  });
  window.dispatchEvent(new Event("resize"));
}

function Probe() {
  const showsMobilePanelPage = useConversationPanelRoute(CONVERSATION_ID);
  const location = useLocation();
  return (
    <div>
      <span data-testid="path">{location.pathname}</span>
      <span data-testid="mobile-panel-page">
        {String(showsMobilePanelPage)}
      </span>
    </div>
  );
}

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/conversations/:conversationId" element={<Probe />} />
        <Route
          path="/conversations/:conversationId/panel"
          element={<Probe />}
        />
      </Routes>
    </MemoryRouter>,
  );
}

function openTerminalFromAgent() {
  act(() => {
    handleCanvasUIAction(
      {
        kind: "CanvasUIAction",
        command: "open_tab",
        tab: "terminal",
      } as CanvasUIAction,
      CONVERSATION_ID,
    );
  });
}

describe("useConversationPanelRoute", () => {
  const originalInnerWidth = window.innerWidth;

  beforeEach(() => {
    localStorage.clear();
    useConversationStore.setState({
      selectedTab: "files",
      isRightPanelShown: false,
      hasRightPanelToggled: false,
    });
  });

  afterEach(() => {
    setWindowWidth(originalInnerWidth);
  });

  it("opens the panel page at phone width when the agent reveals a tab", () => {
    setWindowWidth(PHONE_WIDTH);
    renderAt(CHAT_PATH);

    openTerminalFromAgent();

    expect(screen.getByTestId("path")).toHaveTextContent(PANEL_PATH);
    expect(screen.getByTestId("mobile-panel-page")).toHaveTextContent("true");
    expect(useConversationStore.getState().selectedTab).toBe("terminal");
  });

  it("keeps the desktop chat route and opens the side drawer instead", () => {
    setWindowWidth(DESKTOP_WIDTH);
    renderAt(CHAT_PATH);

    openTerminalFromAgent();

    expect(screen.getByTestId("path")).toHaveTextContent(CHAT_PATH);
    expect(screen.getByTestId("path")).not.toHaveTextContent("/panel");
    expect(useConversationStore.getState().isRightPanelShown).toBe(true);
  });

  it("keeps the full-screen panel page at phone width", () => {
    setWindowWidth(PHONE_WIDTH);
    renderAt(PANEL_PATH);

    expect(screen.getByTestId("path")).toHaveTextContent(PANEL_PATH);
    expect(screen.getByTestId("mobile-panel-page")).toHaveTextContent("true");
  });

  it("falls back to the split layout with the drawer open when the panel page widens to desktop", () => {
    setWindowWidth(PHONE_WIDTH);
    renderAt(PANEL_PATH);

    act(() => setWindowWidth(DESKTOP_WIDTH));

    expect(screen.getByTestId("path")).not.toHaveTextContent("/panel");
    expect(screen.getByTestId("path")).toHaveTextContent(CHAT_PATH);
    expect(screen.getByTestId("mobile-panel-page")).toHaveTextContent("false");
    expect(useConversationStore.getState().isRightPanelShown).toBe(true);
  });
});
