import { describe, it, expect, afterEach, vi } from "vitest";
import { screen, render } from "@testing-library/react";
import React from "react";

// Mock modules before importing the component
vi.mock("react-router", async () => {
  const actual = await vi.importActual("react-router");
  return {
    ...(actual as object),
    useParams: () => ({ conversationId: "test-conversation-id" }),
  };
});

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
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("renders a message if no screenshotSrc is provided", () => {
    useBrowserStore.setState({
      url: "https://example.com",
      screenshotSrc: "",
      reset: vi.fn(),
    });

    render(<BrowserPanel />);

    expect(screen.getByText("BROWSER$NO_PAGE_LOADED")).toBeInTheDocument();
  });

  it("renders the url and a screenshot", () => {
    useBrowserStore.setState({
      url: "https://example.com",
      screenshotSrc:
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mN0uGvyHwAFCAJS091fQwAAAABJRU5ErkJggg==",
      reset: vi.fn(),
    });

    render(<BrowserPanel />);

    expect(screen.getByText("https://example.com")).toBeInTheDocument();
    expect(screen.getByAltText("BROWSER$SCREENSHOT_ALT")).toBeInTheDocument();
  });
});
