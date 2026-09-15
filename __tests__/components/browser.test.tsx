import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";
import { screen, render } from "@testing-library/react";
import React from "react";

// Mock modules before importing the component
vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: "test-conversation-id" }),
  useConversationId: () => ({ conversationId: "test-conversation-id" }),
}));

vi.mock("#/contexts/conversation-context", () => ({
  useConversation: () => ({ conversationId: "test-conversation-id" }),
  ConversationProvider: ({ children }: { children: React.ReactNode }) =>
    children,
}));

vi.mock("react-i18next", async () => {
  const actual = await vi.importActual("react-i18next");
  return {
    ...(actual as object),
    useTranslation: () => ({
      t: (key: string) => key,
      i18n: {
        changeLanguage: () => new Promise(() => {}),
      },
    }),
  };
});

import { BrowserPanel } from "#/components/features/browser/browser";
import { useBrowserStore } from "#/stores/browser-store";

describe("Browser", () => {
  beforeEach(() => {
    useBrowserStore.getState().reset();
  });

  afterEach(() => {
    useBrowserStore.getState().reset();
    vi.clearAllMocks();
  });

  it("renders a message if no screenshotSrc is provided", () => {
    useBrowserStore.setState({
      url: "https://example.com",
      screenshotSrc: "",
    });

    render(<BrowserPanel />);

    expect(screen.getByText("BROWSER$NO_PAGE_LOADED")).toBeInTheDocument();
    expect(screen.getByTestId("browser-chrome-bar")).toBeInTheDocument();
    expect(screen.getByTestId("browser-chrome-url")).toHaveTextContent(
      "https://example.com",
    );
  });

  it("keeps the chrome bar height and disables open-in-new-tab when empty", () => {
    useBrowserStore.setState({
      url: "",
      screenshotSrc: "",
    });

    render(<BrowserPanel />);

    expect(screen.getByTestId("browser-chrome-bar")).toHaveClass("min-h-[34px]");
    expect(screen.getByTestId("browser-chrome-url")).toHaveTextContent(
      "BROWSER$URL_PLACEHOLDER",
    );
    expect(
      screen.queryByRole("button", { name: "BUTTON$BACK" }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "BUTTON$OPEN_IN_NEW_TAB" }),
    ).toBeDisabled();
  });

  it("renders the url and a screenshot", () => {
    useBrowserStore.setState({
      url: "https://example.com",
      screenshotSrc:
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mN0uGvyHwAFCAJS091fQwAAAABJRU5ErkJggg==",
    });

    render(<BrowserPanel />);

    expect(screen.getByTestId("browser-chrome-url")).toHaveTextContent(
      "https://example.com",
    );
    expect(screen.getByAltText("BROWSER$SCREENSHOT_ALT")).toBeInTheDocument();
  });

  it("does not clear a preloaded screenshot when the browser tab first mounts", () => {
    const screenshotSrc =
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mN0uGvyHwAFCAJS091fQwAAAABJRU5ErkJggg==";

    useBrowserStore.setState({
      url: "https://example.com",
      screenshotSrc,
    });

    render(<BrowserPanel />);

    expect(useBrowserStore.getState().screenshotSrc).toBe(screenshotSrc);
    expect(screen.getByAltText("BROWSER$SCREENSHOT_ALT")).toBeInTheDocument();
    expect(screen.queryByText("BROWSER$NO_PAGE_LOADED")).not.toBeInTheDocument();
  });
});
