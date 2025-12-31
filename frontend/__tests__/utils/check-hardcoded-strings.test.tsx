import { render, screen } from "@testing-library/react";
import { test, expect, describe, vi } from "vitest";
import { MemoryRouter } from "react-router";
import { InteractiveChatBox } from "#/components/features/chat/interactive-chat-box";
import { renderWithProviders } from "../../test-utils";

// Mock the translation function
vi.mock("react-i18next", async () => {
  const actual = await vi.importActual("react-i18next");
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string) => {
        // Return a mock translation for the test
        const translations: Record<string, string> = {
          CHAT$PLACEHOLDER: "What do you want to build?",
        };
        return translations[key] || key;
      },
    }),
  };
});

// Mock the useActiveConversation hook
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: null,
  }),
}));

// Mock React Router hooks
vi.mock("react-router", async () => {
  const actual = await vi.importActual("react-router");
  return {
    ...actual,
    useNavigate: () => vi.fn(),
    useParams: () => ({ conversationId: "test-conversation-id" }),
  };
});

// Mock other hooks that might be used by the component
vi.mock("#/hooks/use-user-providers", () => ({
  useUserProviders: () => ({
    providers: [],
  }),
}));

vi.mock("#/hooks/use-conversation-name-context-menu", () => ({
  useConversationNameContextMenu: () => ({
    isOpen: false,
    contextMenuRef: { current: null },
    handleContextMenu: vi.fn(),
    handleClose: vi.fn(),
    handleRename: vi.fn(),
    handleDelete: vi.fn(),
  }),
}));

describe("Check for hardcoded English strings", () => {
  test("InteractiveChatBox should not have hardcoded English strings", () => {
    const { container } = renderWithProviders(
      <MemoryRouter>
        <InteractiveChatBox onSubmit={() => {}} />
      </MemoryRouter>,
    );

    // Get all text content
    const text = container.textContent;

    // List of English strings that should be translated
    const hardcodedStrings = [
      "What do you want to build?",
      "Launch from Scratch",
      "Read this",
    ];

    // Check each string
    hardcodedStrings.forEach((str) => {
      expect(text).not.toContain(str);
    });
  });
});
