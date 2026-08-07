import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";
import { screen, render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";

// Mock modules before importing the component
vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: "test-conversation-id" }),
  useConversationId: () => ({ conversationId: "test-conversation-id" }),
}));

vi.mock("#/context/conversation-context", () => ({
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

  it("renders a message if no page is loaded", () => {
    useBrowserStore.setState({
      mode: "empty",
      url: "https://example.com",
      iframeSrc: "",
      screenshotSrc: "",
    });

    render(<BrowserPanel />);

    expect(screen.getByText("BROWSER$SERVER_MESSAGE")).toBeInTheDocument();
    expect(screen.getByTestId("browser-chrome-bar")).toBeInTheDocument();
    expect(screen.getByTestId("browser-chrome-url")).toHaveValue(
      "https://example.com",
    );
  });

  it("keeps the chrome bar height and disables nav when empty", () => {
    useBrowserStore.setState({
      mode: "empty",
      url: "",
      iframeSrc: "",
      screenshotSrc: "",
    });

    render(<BrowserPanel />);

    expect(screen.getByTestId("browser-chrome-bar")).toHaveClass("min-h-[34px]");
    expect(screen.getByTestId("browser-chrome-url")).toHaveAttribute(
      "placeholder",
      "BROWSER$URL_PLACEHOLDER",
    );
    expect(screen.getByTestId("browser-chrome-back")).toBeDisabled();
    expect(screen.getByTestId("browser-chrome-forward")).toBeDisabled();
    expect(screen.getByTestId("browser-chrome-reload")).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "BUTTON$OPEN_IN_NEW_TAB" }),
    ).toBeDisabled();
  });

  it("renders the url and a screenshot in screenshot mode", () => {
    useBrowserStore.setState({
      mode: "screenshot",
      url: "https://example.com",
      iframeSrc: "",
      screenshotSrc:
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mN0uGvyHwAFCAJS091fQwAAAABJRU5ErkJggg==",
    });

    render(<BrowserPanel />);

    expect(screen.getByTestId("browser-chrome-url")).toHaveValue(
      "https://example.com",
    );
    expect(screen.getByAltText("BROWSER$SCREENSHOT_ALT")).toBeInTheDocument();
  });

  it("does not clear a preloaded screenshot when the browser tab first mounts", () => {
    const screenshotSrc =
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mN0uGvyHwAFCAJS091fQwAAAABJRU5ErkJggg==";

    useBrowserStore.setState({
      mode: "screenshot",
      url: "https://example.com",
      iframeSrc: "",
      screenshotSrc,
    });

    render(<BrowserPanel />);

    expect(useBrowserStore.getState().screenshotSrc).toBe(screenshotSrc);
    expect(screen.getByAltText("BROWSER$SCREENSHOT_ALT")).toBeInTheDocument();
    expect(screen.queryByText("BROWSER$SERVER_MESSAGE")).not.toBeInTheDocument();
  });

  it("renders a live iframe when mode is live", () => {
    useBrowserStore.setState({
      mode: "live",
      url: "http://localhost:8089/",
      iframeSrc: "http://localhost:8089/",
      screenshotSrc: "",
      history: ["http://localhost:8089/"],
      historyIndex: 0,
    });

    render(<BrowserPanel />);

    const iframe = screen.getByTestId("browser-live-iframe");
    expect(iframe).toHaveAttribute("src", "http://localhost:8089/");
    expect(iframe).toHaveAttribute(
      "sandbox",
      "allow-scripts allow-same-origin allow-forms allow-popups allow-popups-to-escape-sandbox",
    );
    expect(screen.getByTestId("browser-chrome-open-external")).toHaveAttribute(
      "href",
      "http://localhost:8089/",
    );
    expect(screen.queryByAltText("BROWSER$SCREENSHOT_ALT")).not.toBeInTheDocument();
  });

  it("navigates to a typed URL on Enter and opens a live iframe", async () => {
    const user = userEvent.setup();
    render(<BrowserPanel />);

    const input = screen.getByTestId("browser-chrome-url");
    await user.clear(input);
    await user.type(input, "localhost:8089{Enter}");

    expect(useBrowserStore.getState()).toMatchObject({
      mode: "live",
      url: "http://localhost:8089",
      iframeSrc: "http://localhost:8089",
    });
    expect(screen.getByTestId("browser-live-iframe")).toHaveAttribute(
      "src",
      "http://localhost:8089",
    );
  });

  it("supports back, forward, and reload for live navigation", async () => {
    const user = userEvent.setup();
    render(<BrowserPanel />);

    const input = screen.getByTestId("browser-chrome-url");
    await user.clear(input);
    await user.type(input, "localhost:8089{Enter}");
    await user.clear(input);
    await user.type(input, "localhost:8089/page-2{Enter}");

    expect(screen.getByTestId("browser-chrome-back")).toBeEnabled();
    expect(screen.getByTestId("browser-chrome-forward")).toBeEnabled();
    expect(screen.getByTestId("browser-chrome-reload")).toBeEnabled();

    await user.click(screen.getByTestId("browser-chrome-back"));
    expect(useBrowserStore.getState().url).toBe("http://localhost:8089");
    expect(useBrowserStore.getState().iframeSrc).toBe("http://localhost:8089");

    await user.click(screen.getByTestId("browser-chrome-forward"));
    expect(useBrowserStore.getState().url).toBe("http://localhost:8089/page-2");
    expect(useBrowserStore.getState().iframeSrc).toBe(
      "http://localhost:8089/page-2",
    );

    // Reload prefers the iframe browsing context; asserting the click is
    // enough here (jsdom may expose contentWindow.location.reload).
    await user.click(screen.getByTestId("browser-chrome-reload"));
    expect(screen.getByTestId("browser-live-iframe")).toBeInTheDocument();
  });
});
