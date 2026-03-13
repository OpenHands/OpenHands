import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { EnterpriseBanner } from "#/components/features/device-verify/enterprise-banner";

const mockCapture = vi.fn();
vi.mock("posthog-js/react", () => ({
  usePostHog: () => ({
    capture: mockCapture,
  }),
}));

describe("EnterpriseBanner", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal("open", vi.fn());
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  describe("Rendering", () => {
    it("should render the self-hosted label", () => {
      renderWithProviders(<EnterpriseBanner />);

      expect(screen.getByText("ENTERPRISE$SELF_HOSTED")).toBeInTheDocument();
    });

    it("should render the enterprise title", () => {
      renderWithProviders(<EnterpriseBanner />);

      expect(screen.getByText("ENTERPRISE$TITLE")).toBeInTheDocument();
    });

    it("should render the enterprise description", () => {
      renderWithProviders(<EnterpriseBanner />);

      expect(screen.getByText("ENTERPRISE$DESCRIPTION")).toBeInTheDocument();
    });

    it("should render all four enterprise feature items", () => {
      renderWithProviders(<EnterpriseBanner />);

      expect(
        screen.getByText("ENTERPRISE$FEATURE_DATA_PRIVACY"),
      ).toBeInTheDocument();
      expect(
        screen.getByText("ENTERPRISE$FEATURE_DEPLOYMENT"),
      ).toBeInTheDocument();
      expect(screen.getByText("ENTERPRISE$FEATURE_SSO")).toBeInTheDocument();
      expect(
        screen.getByText("ENTERPRISE$FEATURE_SUPPORT"),
      ).toBeInTheDocument();
    });

    it("should render the learn more button", () => {
      renderWithProviders(<EnterpriseBanner />);

      const button = screen.getByRole("button", {
        name: "ENTERPRISE$LEARN_MORE_ARIA",
      });
      expect(button).toBeInTheDocument();
      expect(button).toHaveTextContent("ENTERPRISE$LEARN_MORE");
    });
  });

  describe("Learn More Button Interaction", () => {
    it("should capture PostHog event when learn more button is clicked", async () => {
      const user = userEvent.setup();
      renderWithProviders(<EnterpriseBanner />);

      const button = screen.getByRole("button", {
        name: "ENTERPRISE$LEARN_MORE_ARIA",
      });
      await user.click(button);

      expect(mockCapture).toHaveBeenCalledWith("saas_selfhosted_inquiry");
    });

    it("should open enterprise page in new tab when learn more button is clicked", async () => {
      const user = userEvent.setup();
      renderWithProviders(<EnterpriseBanner />);

      const button = screen.getByRole("button", {
        name: "ENTERPRISE$LEARN_MORE_ARIA",
      });
      await user.click(button);

      expect(window.open).toHaveBeenCalledWith(
        "https://openhands.dev/enterprise",
        "_blank",
        "noopener",
      );
    });

    it("should capture PostHog event before opening the URL", async () => {
      const user = userEvent.setup();
      const callOrder: string[] = [];

      mockCapture.mockImplementation(() => {
        callOrder.push("capture");
      });
      vi.stubGlobal(
        "open",
        vi.fn(() => {
          callOrder.push("open");
        }),
      );

      renderWithProviders(<EnterpriseBanner />);

      const button = screen.getByRole("button", {
        name: "ENTERPRISE$LEARN_MORE_ARIA",
      });
      await user.click(button);

      expect(callOrder).toEqual(["capture", "open"]);
    });
  });
});
