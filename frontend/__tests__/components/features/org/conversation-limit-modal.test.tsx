import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { ConversationLimitModal } from "#/components/features/org/conversation-limit-modal";
import { DEFAULT_CONCURRENT_SANDBOX_LIMIT } from "#/utils/constants";

vi.mock("react-i18next", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-i18next")>()),
  useTranslation: () => ({
    t: (key: string, options?: { limit?: number }) => {
      if (key === "CONVERSATION_LIMIT$DESCRIPTION" && options?.limit) {
        return `Description with limit ${options.limit}`;
      }
      return key;
    },
    i18n: {
      changeLanguage: vi.fn(),
    },
  }),
}));

describe("ConversationLimitModal", () => {
  const onCloseMock = vi.fn();

  const renderModal = (props: {
    onClose?: () => void;
    limit?: number;
  } = {}) => {
    const user = userEvent.setup();
    renderWithProviders(
      <ConversationLimitModal
        onClose={props.onClose ?? onCloseMock}
        limit={props.limit}
      />,
    );
    return { user };
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("Rendering", () => {
    it("should render the modal with correct test id", () => {
      renderModal();

      expect(screen.getByTestId("conversation-limit-modal")).toBeInTheDocument();
    });

    it("should display the title", () => {
      renderModal();

      expect(screen.getByText("CONVERSATION_LIMIT$TITLE")).toBeInTheDocument();
    });

    it("should display the close button", () => {
      renderModal();

      expect(
        screen.getByTestId("conversation-limit-close-button"),
      ).toBeInTheDocument();
    });

    it("should not display a learn more button", () => {
      renderModal();

      expect(
        screen.queryByTestId("conversation-limit-learn-more-button"),
      ).not.toBeInTheDocument();
    });

    it("should display description with provided limit", () => {
      renderModal({ limit: 5 });

      expect(screen.getByText("Description with limit 5")).toBeInTheDocument();
    });

    it("should use default limit when not provided", () => {
      renderModal();

      expect(
        screen.getByText(`Description with limit ${DEFAULT_CONCURRENT_SANDBOX_LIMIT}`),
      ).toBeInTheDocument();
    });
  });

  describe("Close Button", () => {
    it("should call onClose when close button is clicked", async () => {
      const { user } = renderModal();

      const closeButton = screen.getByTestId("conversation-limit-close-button");
      await user.click(closeButton);

      expect(onCloseMock).toHaveBeenCalledTimes(1);
    });
  });

  describe("Different Limit Values", () => {
    it("should display limit of 1", () => {
      renderModal({ limit: 1 });
      expect(screen.getByText("Description with limit 1")).toBeInTheDocument();
    });

    it("should display limit of 10", () => {
      renderModal({ limit: 10 });
      expect(screen.getByText("Description with limit 10")).toBeInTheDocument();
    });

    it("should display limit of 100", () => {
      renderModal({ limit: 100 });
      expect(screen.getByText("Description with limit 100")).toBeInTheDocument();
    });
  });
});
