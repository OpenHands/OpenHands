import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { createRoutesStub } from "react-router";
import { useState } from "react";
import {
  InformationRequestForm,
  RequestType,
} from "#/components/features/onboarding/information-request-form";
import { EnterpriseFormData } from "#/utils/local-storage";

// Mock useTracking
const mockTrackEnterpriseLeadFormSubmitted = vi.fn();
vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackEnterpriseLeadFormSubmitted: mockTrackEnterpriseLeadFormSubmitted,
  }),
}));

const mockOnBack = vi.fn();

// Wrapper to manage form state (needed since component is controlled)
function StatefulForm({ requestType }: { requestType: RequestType }) {
  const [formData, setFormData] = useState<EnterpriseFormData>({ name: "", company: "", email: "", message: "" });
  return <InformationRequestForm requestType={requestType} formData={formData} onFormDataChange={setFormData} onBack={mockOnBack} />;
}

describe("InformationRequestForm", () => {
  const defaultProps = {
    requestType: "saas" as RequestType,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockOnBack.mockClear();
  });

  const renderWithRouter = (props = defaultProps) => {
    const Stub = createRoutesStub([
      {
        path: "/",
        Component: () => <StatefulForm {...props} />,
      },
      {
        path: "/login",
        Component: () => <div data-testid="login-page" />,
      },
      {
        path: "/information-request",
        Component: () => <div data-testid="information-request-page" />,
      },
    ]);

    return render(<Stub initialEntries={["/"]} />);
  };

  it("should render the form", () => {
    renderWithRouter();

    expect(screen.getByTestId("information-request-form")).toBeInTheDocument();
  });

  it("should render the logo", () => {
    renderWithRouter();

    const logo = screen.getByTestId("information-request-form").querySelector("svg");
    expect(logo).toBeInTheDocument();
  });

  it("should render all form fields", () => {
    renderWithRouter();

    expect(screen.getByTestId("form-input-name")).toBeInTheDocument();
    expect(screen.getByTestId("form-input-company")).toBeInTheDocument();
    expect(screen.getByTestId("form-input-email")).toBeInTheDocument();
    expect(screen.getByTestId("form-input-message")).toBeInTheDocument();
  });

  it("should render SaaS-specific title when requestType is saas", () => {
    renderWithRouter({ ...defaultProps, requestType: "saas" });

    expect(screen.getByText("ENTERPRISE$FORM_SAAS_TITLE")).toBeInTheDocument();
  });

  it("should render Self-hosted-specific title when requestType is self-hosted", () => {
    renderWithRouter({ ...defaultProps, requestType: "self-hosted" });

    expect(screen.getByText("ENTERPRISE$FORM_SELF_HOSTED_TITLE")).toBeInTheDocument();
  });

  it("should render cloud icon for SaaS request type", () => {
    renderWithRouter({ ...defaultProps, requestType: "saas" });

    // The card should contain the cloud icon
    const card = screen.getByText("ENTERPRISE$SAAS_TITLE").closest("div");
    expect(card).toBeInTheDocument();
  });

  it("should render stacked icon for self-hosted request type", () => {
    renderWithRouter({ ...defaultProps, requestType: "self-hosted" });

    // The card should contain the stacked icon
    const card = screen.getByText("ENTERPRISE$SELF_HOSTED_TITLE").closest("div");
    expect(card).toBeInTheDocument();
  });

  it("should call onBack when back button is clicked", async () => {
    const user = userEvent.setup();

    renderWithRouter();

    const backButton = screen.getByRole("button", { name: "COMMON$BACK" });
    await user.click(backButton);

    expect(mockOnBack).toHaveBeenCalledTimes(1);
  });

  it("should update form fields when user types", async () => {
    const user = userEvent.setup();
    renderWithRouter();

    const nameInput = screen.getByTestId("form-input-name");
    await user.type(nameInput, "John Doe");

    expect(nameInput).toHaveValue("John Doe");
  });

  it("should update email field when user types", async () => {
    const user = userEvent.setup();
    renderWithRouter();

    const emailInput = screen.getByTestId("form-input-email");
    await user.type(emailInput, "john@example.com");

    expect(emailInput).toHaveValue("john@example.com");
  });

  it("should render message as textarea", () => {
    renderWithRouter();

    const messageInput = screen.getByTestId("form-input-message");
    expect(messageInput.tagName).toBe("TEXTAREA");
  });

  it("should have all fields marked as required", () => {
    renderWithRouter();

    expect(screen.getByTestId("form-input-name")).toBeRequired();
    expect(screen.getByTestId("form-input-company")).toBeRequired();
    expect(screen.getByTestId("form-input-email")).toBeRequired();
    expect(screen.getByTestId("form-input-message")).toBeRequired();
  });

  it("should render submit button", () => {
    renderWithRouter();

    const submitButton = screen.getByRole("button", { name: "ENTERPRISE$FORM_SUBMIT" });
    expect(submitButton).toBeInTheDocument();
    expect(submitButton).toHaveAttribute("type", "submit");
  });

  it("should render back button", () => {
    renderWithRouter();

    const backButton = screen.getByRole("button", { name: "COMMON$BACK" });
    expect(backButton).toBeInTheDocument();
    expect(backButton).toHaveAttribute("type", "button");
  });

  it("should have button group with role and aria-label", () => {
    renderWithRouter();

    const buttonGroup = screen.getByRole("group", { name: "Form actions" });
    expect(buttonGroup).toBeInTheDocument();
  });

  it("should display SaaS card description for saas request type", () => {
    renderWithRouter({ ...defaultProps, requestType: "saas" });

    expect(screen.getByText("ENTERPRISE$SAAS_DESCRIPTION")).toBeInTheDocument();
  });

  it("should display Self-hosted card description for self-hosted request type", () => {
    renderWithRouter({ ...defaultProps, requestType: "self-hosted" });

    expect(screen.getByText("ENTERPRISE$SELF_HOSTED_DESCRIPTION")).toBeInTheDocument();
  });

  describe("form validation", () => {
    it("should not show error state before form submission", () => {
      renderWithRouter();

      const nameInput = screen.getByTestId("form-input-name");
      const companyInput = screen.getByTestId("form-input-company");
      const emailInput = screen.getByTestId("form-input-email");
      const messageInput = screen.getByTestId("form-input-message");

      expect(nameInput).toHaveAttribute("aria-invalid", "false");
      expect(companyInput).toHaveAttribute("aria-invalid", "false");
      expect(emailInput).toHaveAttribute("aria-invalid", "false");
      expect(messageInput).toHaveAttribute("aria-invalid", "false");
    });

    it("should not navigate when form is submitted with empty fields", async () => {
      const user = userEvent.setup();
      renderWithRouter();

      const submitButton = screen.getByRole("button", {
        name: "ENTERPRISE$FORM_SUBMIT",
      });
      await user.click(submitButton);

      // Should stay on form page, not navigate to login
      expect(screen.getByTestId("information-request-form")).toBeInTheDocument();
      expect(screen.queryByTestId("login-page")).not.toBeInTheDocument();
    });

    it("should not call tracking when form is submitted with empty fields", async () => {
      const user = userEvent.setup();
      renderWithRouter();

      const submitButton = screen.getByRole("button", { name: "ENTERPRISE$FORM_SUBMIT" });
      await user.click(submitButton);

      expect(mockTrackEnterpriseLeadFormSubmitted).not.toHaveBeenCalled();
    });

    it("should navigate to login page when form is submitted with all fields filled", async () => {
      const user = userEvent.setup();
      renderWithRouter();

      await user.type(screen.getByTestId("form-input-name"), "John Doe");
      await user.type(screen.getByTestId("form-input-company"), "Acme Inc");
      await user.type(screen.getByTestId("form-input-email"), "john@example.com");
      await user.type(screen.getByTestId("form-input-message"), "Hello world");

      const submitButton = screen.getByRole("button", {
        name: "ENTERPRISE$FORM_SUBMIT",
      });
      await user.click(submitButton);

      // Should navigate to login page
      expect(screen.getByTestId("login-page")).toBeInTheDocument();
    });

    it("should call tracking with form data when form is submitted successfully", async () => {
      const user = userEvent.setup();
      renderWithRouter({ ...defaultProps, requestType: "saas" });

      await user.type(screen.getByTestId("form-input-name"), "John Doe");
      await user.type(screen.getByTestId("form-input-company"), "Acme Inc");
      await user.type(screen.getByTestId("form-input-email"), "john@example.com");
      await user.type(screen.getByTestId("form-input-message"), "Hello world");

      const submitButton = screen.getByRole("button", { name: "ENTERPRISE$FORM_SUBMIT" });
      await user.click(submitButton);

      expect(mockTrackEnterpriseLeadFormSubmitted).toHaveBeenCalledTimes(1);
      expect(mockTrackEnterpriseLeadFormSubmitted).toHaveBeenCalledWith({
        requestType: "saas",
        name: "John Doe",
        company: "Acme Inc",
        email: "john@example.com",
        message: "Hello world",
      });
    });

    it("should call tracking with self-hosted request type", async () => {
      const user = userEvent.setup();
      renderWithRouter({ ...defaultProps, requestType: "self-hosted" });

      await user.type(screen.getByTestId("form-input-name"), "Jane Smith");
      await user.type(screen.getByTestId("form-input-company"), "Tech Corp");
      await user.type(screen.getByTestId("form-input-email"), "jane@techcorp.com");
      await user.type(screen.getByTestId("form-input-message"), "Interested in self-hosted");

      const submitButton = screen.getByRole("button", { name: "ENTERPRISE$FORM_SUBMIT" });
      await user.click(submitButton);

      expect(mockTrackEnterpriseLeadFormSubmitted).toHaveBeenCalledWith({
        requestType: "self-hosted",
        name: "Jane Smith",
        company: "Tech Corp",
        email: "jane@techcorp.com",
        message: "Interested in self-hosted",
      });
    });

    it("should trim whitespace from form fields before tracking", async () => {
      const user = userEvent.setup();
      renderWithRouter();

      await user.type(screen.getByTestId("form-input-name"), "  John Doe  ");
      await user.type(screen.getByTestId("form-input-company"), "  Acme Inc  ");
      await user.type(screen.getByTestId("form-input-email"), "  john@example.com  ");
      await user.type(screen.getByTestId("form-input-message"), "  Hello world  ");

      const submitButton = screen.getByRole("button", { name: "ENTERPRISE$FORM_SUBMIT" });
      await user.click(submitButton);

      expect(mockTrackEnterpriseLeadFormSubmitted).toHaveBeenCalledWith({
        requestType: "saas",
        name: "John Doe",
        company: "Acme Inc",
        email: "john@example.com",
        message: "Hello world",
      });
    });

    it("should have valid aria-invalid state when field has value", async () => {
      const user = userEvent.setup();
      renderWithRouter();

      const nameInput = screen.getByTestId("form-input-name");
      await user.type(nameInput, "John Doe");

      // Field with value should not be invalid
      expect(nameInput).toHaveAttribute("aria-invalid", "false");
    });

    it("should not navigate when email is invalid", async () => {
      const user = userEvent.setup();
      renderWithRouter();

      await user.type(screen.getByTestId("form-input-name"), "John Doe");
      await user.type(screen.getByTestId("form-input-company"), "Acme Inc");
      await user.type(screen.getByTestId("form-input-email"), "invalid-email");
      await user.type(screen.getByTestId("form-input-message"), "Hello world");

      const submitButton = screen.getByRole("button", {
        name: "ENTERPRISE$FORM_SUBMIT",
      });
      await user.click(submitButton);

      // Should stay on form page, not navigate to login
      expect(screen.getByTestId("information-request-form")).toBeInTheDocument();
      expect(screen.queryByTestId("login-page")).not.toBeInTheDocument();
      expect(mockTrackEnterpriseLeadFormSubmitted).not.toHaveBeenCalled();
    });
  });

  describe("loading state", () => {
    it("should prevent double submission", async () => {
      const user = userEvent.setup();
      renderWithRouter();

      await user.type(screen.getByTestId("form-input-name"), "John Doe");
      await user.type(screen.getByTestId("form-input-company"), "Acme Inc");
      await user.type(screen.getByTestId("form-input-email"), "john@example.com");
      await user.type(screen.getByTestId("form-input-message"), "Hello world");

      const submitButton = screen.getByRole("button", {
        name: "ENTERPRISE$FORM_SUBMIT",
      });

      // Click multiple times rapidly
      await user.click(submitButton);
      await user.click(submitButton);
      await user.click(submitButton);

      // Should only track once
      expect(mockTrackEnterpriseLeadFormSubmitted).toHaveBeenCalledTimes(1);
      // Should navigate to login page
      expect(screen.getByTestId("login-page")).toBeInTheDocument();
    });
  });
});
