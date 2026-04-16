import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createRoutesStub } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { I18nextProvider } from "react-i18next";
import i18n from "i18next";
import OnboardingForm, { clientLoader } from "#/routes/onboarding-form";

const mockMutate = vi.fn();
const mockNavigate = vi.fn();
const mockUseMe = vi.fn();
const mockTrackOnboardingCompleted = vi.fn();

// Loader data set in beforeEach for each test suite
let loaderData: { config: { app_mode: string; feature_flags: { deployment_mode: string } } };

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

vi.mock("#/hooks/query/use-me", () => ({
  useMe: () => mockUseMe(),
}));

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackOnboardingCompleted: mockTrackOnboardingCompleted,
  }),
}));

// Mocks for clientLoader tests
const mockQueryClientGetData = vi.fn();
const mockQueryClientSetData = vi.fn();
vi.mock("#/query-client-config", () => ({
  queryClient: {
    getQueryData: (...args: unknown[]) => mockQueryClientGetData(...args),
    setQueryData: (...args: unknown[]) => mockQueryClientSetData(...args),
  },
}));

const mockGetConfig = vi.fn();
vi.mock("#/api/option-service/option-service.api", () => ({
  default: {
    getConfig: () => mockGetConfig(),
  },
}));

const renderOnboardingForm = async () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  const RouterStub = createRoutesStub([
    {
      path: "/",
      Component: OnboardingForm,
      loader: () => loaderData,
    },
  ]);

  const result = render(
    <I18nextProvider i18n={i18n}>
      <QueryClientProvider client={queryClient}>
        <RouterStub initialEntries={["/"]} />
      </QueryClientProvider>
    </I18nextProvider>,
  );

  // Wait for the component to render
  await screen.findByTestId("onboarding-form");
  return result;
};

describe("OnboardingForm - Cloud Mode", () => {
  beforeEach(() => {
    mockMutate.mockClear();
    mockNavigate.mockClear();
    mockTrackOnboardingCompleted.mockClear();
    loaderData = {
      config: {
        app_mode: "saas",
        feature_flags: { deployment_mode: "cloud" },
      },
    };
    // Cloud mode tracks all users, role doesn't matter
    mockUseMe.mockReturnValue({ data: { role: "member" } });
  });

  it("should render with the correct test id", async () => {
    await renderOnboardingForm();

    expect(screen.getByTestId("onboarding-form")).toBeInTheDocument();
  });

  it("should render the first step initially", async () => {
    await renderOnboardingForm();

    expect(screen.getByTestId("step-header")).toBeInTheDocument();
    expect(screen.getByTestId("step-content")).toBeInTheDocument();
    expect(screen.getByTestId("step-actions")).toBeInTheDocument();
  });

  it("should display step progress indicator with 3 bars for cloud mode", async () => {
    await renderOnboardingForm();

    const stepHeader = screen.getByTestId("step-header");
    const progressBars = stepHeader.querySelectorAll(".rounded-full");
    expect(progressBars).toHaveLength(3);
  });

  it("should have the Next button disabled when no option is selected", async () => {
    await renderOnboardingForm();

    const nextButton = screen.getByRole("button", { name: /next/i });
    expect(nextButton).toBeDisabled();
  });

  it("should enable the Next button when an option is selected", async () => {
    const user = userEvent.setup();
    await renderOnboardingForm();

    await user.click(screen.getByTestId("step-option-solo"));

    const nextButton = screen.getByRole("button", { name: /next/i });
    expect(nextButton).not.toBeDisabled();
  });

  it("should advance to the next step when Next is clicked", async () => {
    const user = userEvent.setup();
    await renderOnboardingForm();

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
    await renderOnboardingForm();

    await user.click(screen.getByTestId("step-option-solo"));
    await user.click(screen.getByRole("button", { name: /next/i }));

    const nextButton = screen.getByRole("button", { name: /next/i });
    expect(nextButton).toBeDisabled();
  });

  it("should call submitOnboarding with selections when finishing the last step", async () => {
    const user = userEvent.setup();
    await renderOnboardingForm();

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

  it("should track onboarding completion to PostHog in cloud mode", async () => {
    const user = userEvent.setup();
    await renderOnboardingForm();

    // Complete the full cloud onboarding flow
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

  it("should render 5 options on step 1 (org size question)", async () => {
    await renderOnboardingForm();

    const options = screen
      .getAllByRole("button")
      .filter((btn) =>
        btn.getAttribute("data-testid")?.startsWith("step-option-"),
      );
    expect(options).toHaveLength(5);
  });

  it("should preserve selections when navigating through steps", async () => {
    const user = userEvent.setup();
    await renderOnboardingForm();

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
    await renderOnboardingForm();

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
    await renderOnboardingForm();

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
    await renderOnboardingForm();

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

  it("should not render the Back button on the first step", async () => {
    await renderOnboardingForm();

    const backButton = screen.queryByRole("button", { name: /back/i });
    expect(backButton).not.toBeInTheDocument();
  });

  it("should render the Back button on step 2", async () => {
    const user = userEvent.setup();
    await renderOnboardingForm();

    await user.click(screen.getByTestId("step-option-solo"));
    await user.click(screen.getByRole("button", { name: /next/i }));

    const backButton = screen.getByRole("button", { name: /back/i });
    expect(backButton).toBeInTheDocument();
  });

  it("should go back to the previous step when Back is clicked", async () => {
    const user = userEvent.setup();
    await renderOnboardingForm();

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

describe("OnboardingForm - Self-Hosted Mode", () => {
  // Self-hosted mode has 3 steps: org_name, org_size, use_case
  // The role question is cloud-only and not shown in self-hosted mode

  beforeEach(() => {
    mockMutate.mockClear();
    mockNavigate.mockClear();
    mockTrackOnboardingCompleted.mockClear();
    loaderData = {
      config: {
        app_mode: "saas",
        feature_flags: { deployment_mode: "self_hosted" },
      },
    };
    // Self-hosted mode only tracks org owners
    mockUseMe.mockReturnValue({ data: { role: "owner" } });
  });

  it("should render with the correct test id", async () => {
    await renderOnboardingForm();

    expect(screen.getByTestId("onboarding-form")).toBeInTheDocument();
  });

  it("should display step progress indicator with 3 bars for self-hosted mode", async () => {
    await renderOnboardingForm();

    // Self-hosted has 3 steps: org_name, org_size, use_case (role is cloud-only)
    const stepHeader = screen.getByTestId("step-header");
    const progressBars = stepHeader.querySelectorAll(".rounded-full");
    expect(progressBars).toHaveLength(3);
  });

  it("should start with org_name question as first step with two input fields", async () => {
    await renderOnboardingForm();

    // The first step in self-hosted mode should be org_name with two inputs
    const orgNameInput = screen.getByTestId("form-input-org_name");
    const orgDomainInput = screen.getByTestId("form-input-org_domain");
    expect(orgNameInput).toBeInTheDocument();
    expect(orgDomainInput).toBeInTheDocument();
  });

  it("should call submitOnboarding with all selections including org_name when finishing", async () => {
    const user = userEvent.setup();
    await renderOnboardingForm();

    // Step 1 - enter org name and domain (input fields)
    const orgNameInput = screen.getByTestId("form-input-org_name");
    const orgDomainInput = screen.getByTestId("form-input-org_domain");
    await user.type(orgNameInput, "Acme Corp");
    await user.type(orgDomainInput, "acme.com");
    await user.click(screen.getByRole("button", { name: /next/i }));

    // Step 2 - select org size (single select)
    await user.click(screen.getByTestId("step-option-org_2_10"));
    await user.click(screen.getByRole("button", { name: /next/i }));

    // Step 3 - select use case (multi-select) - this is the last step in self-hosted mode
    await user.click(screen.getByTestId("step-option-new_features"));
    await user.click(screen.getByRole("button", { name: /finish/i }));

    expect(mockMutate).toHaveBeenCalledTimes(1);
    expect(mockMutate).toHaveBeenCalledWith({
      selections: {
        org_name: "Acme Corp",
        org_domain: "acme.com",
        org_size: "org_2_10",
        use_case: ["new_features"],
      },
    });
  });

  it("should track onboarding completion in self-hosted mode", async () => {
    const user = userEvent.setup();
    await renderOnboardingForm();

    // Complete the full self-hosted onboarding flow (3 steps)
    const orgNameInput = screen.getByTestId("form-input-org_name");
    const orgDomainInput = screen.getByTestId("form-input-org_domain");
    await user.type(orgNameInput, "Test Company");
    await user.type(orgDomainInput, "test.com");
    await user.click(screen.getByRole("button", { name: /next/i }));

    await user.click(screen.getByTestId("step-option-org_2_10"));
    await user.click(screen.getByRole("button", { name: /next/i }));

    await user.click(screen.getByTestId("step-option-new_features"));
    await user.click(screen.getByRole("button", { name: /finish/i }));

    expect(mockTrackOnboardingCompleted).toHaveBeenCalledTimes(1);
    // Note: role is not included since role question is cloud-only
    expect(mockTrackOnboardingCompleted).toHaveBeenCalledWith({
      role: undefined,
      orgSize: "org_2_10",
      useCase: ["new_features"],
    });
  });

  it("should show all 3 progress bars filled on the last step", async () => {
    const user = userEvent.setup();
    await renderOnboardingForm();

    // Navigate through all 3 steps
    const orgNameInput = screen.getByTestId("form-input-org_name");
    const orgDomainInput = screen.getByTestId("form-input-org_domain");
    await user.type(orgNameInput, "Test Company");
    await user.type(orgDomainInput, "test.com");
    await user.click(screen.getByRole("button", { name: /next/i }));

    await user.click(screen.getByTestId("step-option-org_2_10"));
    await user.click(screen.getByRole("button", { name: /next/i }));

    // On step 3, all three progress bars should be filled
    const stepHeader = screen.getByTestId("step-header");
    const progressBars = stepHeader.querySelectorAll(".bg-white");
    expect(progressBars).toHaveLength(3);
  });

  it("should have Next button disabled when both org_name inputs are empty", async () => {
    await renderOnboardingForm();

    const nextButton = screen.getByRole("button", { name: /next/i });
    expect(nextButton).toBeDisabled();
  });

  it("should enable Next button when both org_name and org_domain are entered", async () => {
    const user = userEvent.setup();
    await renderOnboardingForm();

    const orgNameInput = screen.getByTestId("form-input-org_name");
    const orgDomainInput = screen.getByTestId("form-input-org_domain");
    await user.type(orgNameInput, "My Company");
    await user.type(orgDomainInput, "mycompany.com");

    const nextButton = screen.getByRole("button", { name: /next/i });
    expect(nextButton).not.toBeDisabled();
  });

  it("should NOT track onboarding completion for non-owners in self-hosted mode", async () => {
    // Override the mock to return a member (non-owner) role
    mockUseMe.mockReturnValue({ data: { role: "member" } });

    const user = userEvent.setup();
    await renderOnboardingForm();

    // Complete the full self-hosted onboarding flow (3 steps)
    const orgNameInput = screen.getByTestId("form-input-org_name");
    const orgDomainInput = screen.getByTestId("form-input-org_domain");
    await user.type(orgNameInput, "Test Company");
    await user.type(orgDomainInput, "test.com");
    await user.click(screen.getByRole("button", { name: /next/i }));

    await user.click(screen.getByTestId("step-option-org_2_10"));
    await user.click(screen.getByRole("button", { name: /next/i }));

    await user.click(screen.getByTestId("step-option-new_features"));
    await user.click(screen.getByRole("button", { name: /finish/i }));

    // Tracking should NOT be called for non-owners in self-hosted mode
    expect(mockTrackOnboardingCompleted).not.toHaveBeenCalled();

    // But onboarding submission should still work
    expect(mockMutate).toHaveBeenCalledTimes(1);
  });

  it("should NOT track onboarding completion for admins in self-hosted mode", async () => {
    // Override the mock to return an admin role
    mockUseMe.mockReturnValue({ data: { role: "admin" } });

    const user = userEvent.setup();
    await renderOnboardingForm();

    // Complete the full self-hosted onboarding flow (3 steps)
    const orgNameInput = screen.getByTestId("form-input-org_name");
    const orgDomainInput = screen.getByTestId("form-input-org_domain");
    await user.type(orgNameInput, "Test Company");
    await user.type(orgDomainInput, "test.com");
    await user.click(screen.getByRole("button", { name: /next/i }));

    await user.click(screen.getByTestId("step-option-org_2_10"));
    await user.click(screen.getByRole("button", { name: /next/i }));

    await user.click(screen.getByTestId("step-option-new_features"));
    await user.click(screen.getByRole("button", { name: /finish/i }));

    // Tracking should NOT be called for admins in self-hosted mode (only owners)
    expect(mockTrackOnboardingCompleted).not.toHaveBeenCalled();

    // But onboarding submission should still work
    expect(mockMutate).toHaveBeenCalledTimes(1);
  });
});

describe("onboarding-form clientLoader", () => {
  beforeEach(() => {
    mockQueryClientGetData.mockReset();
    mockQueryClientSetData.mockReset();
    mockGetConfig.mockReset();
  });

  describe("redirect behavior", () => {
    it("should redirect to / when app_mode is oss", async () => {
      const ossConfig = {
        app_mode: "oss",
        feature_flags: { deployment_mode: undefined },
      };
      mockQueryClientGetData.mockReturnValue(ossConfig);

      const result = await clientLoader();

      expect(result).toBeDefined();
      expect((result as Response).status).toBe(302);
      expect((result as Response).headers.get("Location")).toBe("/");
    });

    it("should redirect to / when app_mode is undefined", async () => {
      const undefinedConfig = {
        app_mode: undefined,
        feature_flags: { deployment_mode: "cloud" },
      };
      mockQueryClientGetData.mockReturnValue(undefinedConfig);

      const result = await clientLoader();

      expect(result).toBeDefined();
      expect((result as Response).status).toBe(302);
      expect((result as Response).headers.get("Location")).toBe("/");
    });

    it("should redirect to / when config is null", async () => {
      mockQueryClientGetData.mockReturnValue(null);
      mockGetConfig.mockResolvedValue(null);

      const result = await clientLoader();

      expect(result).toBeDefined();
      expect((result as Response).status).toBe(302);
      expect((result as Response).headers.get("Location")).toBe("/");
    });

    it("should allow access and return config when app_mode is saas with cloud deployment", async () => {
      const saasCloudConfig = {
        app_mode: "saas",
        feature_flags: { deployment_mode: "cloud" },
      };
      mockQueryClientGetData.mockReturnValue(saasCloudConfig);

      const result = await clientLoader();

      expect(result).toEqual({ config: saasCloudConfig });
    });

    it("should allow access and return config when app_mode is saas with self_hosted deployment", async () => {
      const saasSelfHostedConfig = {
        app_mode: "saas",
        feature_flags: { deployment_mode: "self_hosted" },
      };
      mockQueryClientGetData.mockReturnValue(saasSelfHostedConfig);

      const result = await clientLoader();

      expect(result).toEqual({ config: saasSelfHostedConfig });
    });
  });

  describe("config fetching", () => {
    it("should use cached config from queryClient when available", async () => {
      const cachedConfig = {
        app_mode: "saas",
        feature_flags: { deployment_mode: "cloud" },
      };
      mockQueryClientGetData.mockReturnValue(cachedConfig);

      await clientLoader();

      expect(mockQueryClientGetData).toHaveBeenCalledWith(["web-client-config"]);
      expect(mockGetConfig).not.toHaveBeenCalled();
    });

    it("should fetch config from OptionService when not cached", async () => {
      const fetchedConfig = {
        app_mode: "saas",
        feature_flags: { deployment_mode: "cloud" },
      };
      mockQueryClientGetData.mockReturnValue(null);
      mockGetConfig.mockResolvedValue(fetchedConfig);

      const result = await clientLoader();

      expect(mockGetConfig).toHaveBeenCalled();
      expect(mockQueryClientSetData).toHaveBeenCalledWith(
        ["web-client-config"],
        fetchedConfig,
      );
      expect(result).toEqual({ config: fetchedConfig });
    });
  });
});
