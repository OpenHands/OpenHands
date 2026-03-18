import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../../test-utils";
import OnboardingForm from "#/routes/onboarding-form";

const mockMutate = vi.fn();
const mockNavigate = vi.fn();
const mockUseConfig = vi.fn();
const mockTrackOnboardingCompleted = vi.fn();

vi.mock("react-router", async (importOriginal) => {
  const original = await importOriginal<typeof import("react-router")>();
  return {
    ...original,
    useNavigate: () => mockNavigate,
  };
});

vi.mock("#/hooks/mutation/use-submit-onboarding", () => ({
  useSubmitOnboarding: () => ({
    mutate: mockMutate,
  }),
}));

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => mockUseConfig(),
}));

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackOnboardingCompleted: mockTrackOnboardingCompleted,
  }),
}));

const renderOnboardingForm = () => {
  return renderWithProviders(
    <MemoryRouter>
      <OnboardingForm />
    </MemoryRouter>,
  );
};

describe("OnboardingForm - SaaS Mode", () => {
  beforeEach(() => {
    mockMutate.mockClear();
    mockNavigate.mockClear();
    mockTrackOnboardingCompleted.mockClear();
    mockUseConfig.mockReturnValue({
      data: { app_mode: "saas" },
      isLoading: false,
    });
  });

  it("should render with the correct test id", () => {
    renderOnboardingForm();

    expect(screen.getByTestId("onboarding-form")).toBeInTheDocument();
  });

  it("should render the first step initially", () => {
    renderOnboardingForm();

    expect(screen.getByTestId("step-header")).toBeInTheDocument();
    expect(screen.getByTestId("step-content")).toBeInTheDocument();
    expect(screen.getByTestId("step-actions")).toBeInTheDocument();
  });

  it("should display step progress indicator with 3 bars for saas mode", () => {
    renderOnboardingForm();

    const stepHeader = screen.getByTestId("step-header");
    const progressBars = stepHeader.querySelectorAll(".rounded-full");
    expect(progressBars).toHaveLength(3);
  });

  it("should have the Next button disabled when no option is selected", () => {
    renderOnboardingForm();

    const nextButton = screen.getByRole("button", { name: /next/i });
    expect(nextButton).toBeDisabled();
  });

  it("should enable the Next button when an option is selected", async () => {
    const user = userEvent.setup();
    renderOnboardingForm();

    await user.click(screen.getByTestId("step-option-solo"));

    const nextButton = screen.getByRole("button", { name: /next/i });
    expect(nextButton).not.toBeDisabled();
  });

  it("should advance to the next step when Next is clicked", async () => {
    const user = userEvent.setup();
    renderOnboardingForm();

    // On step 1, first progress bar should be filled (bg-white)
    const stepHeader = screen.getByTestId("step-header");
    let progressBars = stepHeader.querySelectorAll(".bg-white");
    expect(progressBars).toHaveLength(1);

    await user.click(screen.getByTestId("step-option-solo"));
    await user.click(screen.getByRole("button", { name: /next/i }));

    // On step 2, first two progress bars should be filled
    progressBars = stepHeader.querySelectorAll(".bg-white");
    expect(progressBars).toHaveLength(2);
  });

  it("should disable Next button again on new step until option is selected", async () => {
    const user = userEvent.setup();
    renderOnboardingForm();

    await user.click(screen.getByTestId("step-option-solo"));
    await user.click(screen.getByRole("button", { name: /next/i }));

    const nextButton = screen.getByRole("button", { name: /next/i });
    expect(nextButton).toBeDisabled();
  });

  it("should call submitOnboarding with selections when finishing the last step", async () => {
    const user = userEvent.setup();
    renderOnboardingForm();

    // Step 1 - select org size (first step in saas mode - single select)
    await user.click(screen.getByTestId("step-option-org_2_10"));
    await user.click(screen.getByRole("button", { name: /next/i }));

    // Step 2 - select use case (multi-select)
    await user.click(screen.getByTestId("step-option-new_features"));
    await user.click(screen.getByRole("button", { name: /next/i }));

    // Step 3 - select role (last step in saas mode - single select)
    await user.click(screen.getByTestId("step-option-software_engineer"));
    await user.click(screen.getByRole("button", { name: /finish/i }));

    expect(mockMutate).toHaveBeenCalledTimes(1);
    expect(mockMutate).toHaveBeenCalledWith({
      selections: {
        org_size: "org_2_10",
        use_case: ["new_features"],
        role: "software_engineer",
      },
    });
  });

  it("should track onboarding completion to PostHog in SaaS mode", async () => {
    const user = userEvent.setup();
    renderOnboardingForm();

    // Complete the full SaaS onboarding flow
    await user.click(screen.getByTestId("step-option-org_2_10"));
    await user.click(screen.getByRole("button", { name: /next/i }));

    await user.click(screen.getByTestId("step-option-new_features"));
    await user.click(screen.getByRole("button", { name: /next/i }));

    await user.click(screen.getByTestId("step-option-software_engineer"));
    await user.click(screen.getByRole("button", { name: /finish/i }));

    expect(mockTrackOnboardingCompleted).toHaveBeenCalledTimes(1);
    expect(mockTrackOnboardingCompleted).toHaveBeenCalledWith({
      role: "software_engineer",
      orgSize: "org_2_10",
      useCase: ["new_features"],
    });
  });

  it("should render 5 options on step 1 (org size question)", () => {
    renderOnboardingForm();

    const options = screen
      .getAllByRole("button")
      .filter((btn) =>
        btn.getAttribute("data-testid")?.startsWith("step-option-"),
      );
    expect(options).toHaveLength(5);
  });

  it("should preserve selections when navigating through steps", async () => {
    const user = userEvent.setup();
    renderOnboardingForm();

    // Select org size on step 1 (single select)
    await user.click(screen.getByTestId("step-option-solo"));
    await user.click(screen.getByRole("button", { name: /next/i }));

    // Select use case on step 2 (multi-select)
    await user.click(screen.getByTestId("step-option-fixing_bugs"));
    await user.click(screen.getByRole("button", { name: /next/i }));

    // Select role on step 3 (single select)
    await user.click(screen.getByTestId("step-option-cto_founder"));
    await user.click(screen.getByRole("button", { name: /finish/i }));

    // Verify all selections were preserved
    expect(mockMutate).toHaveBeenCalledWith({
      selections: {
        org_size: "solo",
        use_case: ["fixing_bugs"],
        role: "cto_founder",
      },
    });
  });

  it("should allow selecting multiple options on multi-select steps", async () => {
    const user = userEvent.setup();
    renderOnboardingForm();

    // Step 1 - select org size (single select)
    await user.click(screen.getByTestId("step-option-solo"));
    await user.click(screen.getByRole("button", { name: /next/i }));

    // Step 2 - select multiple use cases (multi-select)
    await user.click(screen.getByTestId("step-option-new_features"));
    await user.click(screen.getByTestId("step-option-fixing_bugs"));
    await user.click(screen.getByTestId("step-option-refactoring"));
    await user.click(screen.getByRole("button", { name: /next/i }));

    // Step 3 - select role (single select)
    await user.click(screen.getByTestId("step-option-software_engineer"));
    await user.click(screen.getByRole("button", { name: /finish/i }));

    expect(mockMutate).toHaveBeenCalledWith({
      selections: {
        org_size: "solo",
        use_case: ["new_features", "fixing_bugs", "refactoring"],
        role: "software_engineer",
      },
    });
  });

  it("should allow deselecting options on multi-select steps", async () => {
    const user = userEvent.setup();
    renderOnboardingForm();

    // Step 1 - select org size
    await user.click(screen.getByTestId("step-option-solo"));
    await user.click(screen.getByRole("button", { name: /next/i }));

    // Step 2 - select and deselect use cases
    await user.click(screen.getByTestId("step-option-new_features"));
    await user.click(screen.getByTestId("step-option-fixing_bugs"));
    await user.click(screen.getByTestId("step-option-new_features")); // Deselect

    await user.click(screen.getByRole("button", { name: /next/i }));

    // Step 3 - select role
    await user.click(screen.getByTestId("step-option-software_engineer"));
    await user.click(screen.getByRole("button", { name: /finish/i }));

    expect(mockMutate).toHaveBeenCalledWith({
      selections: {
        org_size: "solo",
        use_case: ["fixing_bugs"],
        role: "software_engineer",
      },
    });
  });

  it("should show all progress bars filled on the last step", async () => {
    const user = userEvent.setup();
    renderOnboardingForm();

    // Navigate to step 3
    await user.click(screen.getByTestId("step-option-solo"));
    await user.click(screen.getByRole("button", { name: /next/i }));

    await user.click(screen.getByTestId("step-option-new_features"));
    await user.click(screen.getByRole("button", { name: /next/i }));

    // On step 3, all three progress bars should be filled
    const stepHeader = screen.getByTestId("step-header");
    const progressBars = stepHeader.querySelectorAll(".bg-white");
    expect(progressBars).toHaveLength(3);
  });

  it("should not render the Back button on the first step", () => {
    renderOnboardingForm();

    const backButton = screen.queryByRole("button", { name: /back/i });
    expect(backButton).not.toBeInTheDocument();
  });

  it("should render the Back button on step 2", async () => {
    const user = userEvent.setup();
    renderOnboardingForm();

    await user.click(screen.getByTestId("step-option-solo"));
    await user.click(screen.getByRole("button", { name: /next/i }));

    const backButton = screen.getByRole("button", { name: /back/i });
    expect(backButton).toBeInTheDocument();
  });

  it("should go back to the previous step when Back is clicked", async () => {
    const user = userEvent.setup();
    renderOnboardingForm();

    // Navigate to step 2
    await user.click(screen.getByTestId("step-option-solo"));
    await user.click(screen.getByRole("button", { name: /next/i }));

    // Verify we're on step 2 (2 progress bars filled)
    const stepHeader = screen.getByTestId("step-header");
    let progressBars = stepHeader.querySelectorAll(".bg-white");
    expect(progressBars).toHaveLength(2);

    // Click Back
    await user.click(screen.getByRole("button", { name: /back/i }));

    // Verify we're back on step 1 (1 progress bar filled)
    progressBars = stepHeader.querySelectorAll(".bg-white");
    expect(progressBars).toHaveLength(1);
  });
});
