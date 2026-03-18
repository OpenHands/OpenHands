import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { EnterpriseBanner } from "#/components/features/device-verify/enterprise-banner";

const mockCapture = vi.fn();
vi.mock("posthog-js/react", () => ({
  usePostHog: () => ({
    capture: mockCapture,
  }),
}));

const { ENABLE_PROJ_USER_JOURNEY_MOCK } = vi.hoisted(() => ({
  ENABLE_PROJ_USER_JOURNEY_MOCK: vi.fn(() => true),
}));

vi.mock("#/utils/feature-flags", () => ({
  ENABLE_PROJ_USER_JOURNEY: () => ENABLE_PROJ_USER_JOURNEY_MOCK(),
}));

describe("EnterpriseBanner", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    ENABLE_PROJ_USER_JOURNEY_MOCK.mockReturnValue(true);
  });

  describe("Feature Flag", () => {
    it("should not render when proj_user_journey feature flag is disabled", () => {
      ENABLE_PROJ_USER_JOURNEY_MOCK.mockReturnValue(false);

      const { container } = renderWithProviders(<EnterpriseBanner />);

      expect(container.firstChild).toBeNull();
      expect(screen.queryByText("ENTERPRISE$TITLE")).not.toBeInTheDocument();
    });

    it("should render when proj_user_journey feature flag is enabled", () => {
      ENABLE_PROJ_USER_JOURNEY_MOCK.mockReturnValue(true);

      renderWithProviders(<EnterpriseBanner />);

      expect(screen.getByText("ENTERPRISE$TITLE")).toBeInTheDocument();
    });
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

    it("should render the learn more link", () => {
      renderWithProviders(<EnterpriseBanner />);

      const link = screen.getByRole("link", {
        name: "ENTERPRISE$LEARN_MORE_ARIA",
      });
      expect(link).toBeInTheDocument();
      expect(link).toHaveTextContent("ENTERPRISE$LEARN_MORE");
      expect(link).toHaveAttribute("href", "https://openhands.dev/enterprise");
      expect(link).toHaveAttribute("target", "_blank");
      expect(link).toHaveAttribute("rel", "noopener noreferrer");
    });
  });

  describe("Learn More Link Interaction", () => {
    it("should capture PostHog event when learn more link is clicked", async () => {
      const user = userEvent.setup();
      renderWithProviders(<EnterpriseBanner />);

      const link = screen.getByRole("link", {
        name: "ENTERPRISE$LEARN_MORE_ARIA",
      });
      await user.click(link);

      expect(mockCapture).toHaveBeenCalledWith("saas_selfhosted_inquiry");
    });

    it("should have correct href attribute for opening in new tab", () => {
      renderWithProviders(<EnterpriseBanner />);

      const link = screen.getByRole("link", {
        name: "ENTERPRISE$LEARN_MORE_ARIA",
      });
      expect(link).toHaveAttribute("href", "https://openhands.dev/enterprise");
      expect(link).toHaveAttribute("target", "_blank");
    });
  });
});
