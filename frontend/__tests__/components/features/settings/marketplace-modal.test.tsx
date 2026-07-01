import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { MarketplaceModal } from "#/components/features/settings/marketplace-modal";

const onClose = vi.fn();
const onSave = vi.fn();

const baseProps = {
  isOpen: true,
  mode: "add" as const,
  onClose,
  onSave,
  isSaving: false,
};

beforeEach(() => {
  onClose.mockReset();
  onSave.mockReset();
});

describe("MarketplaceModal", () => {
  it("renders nothing when not open", () => {
    const { container } = render(
      <MarketplaceModal {...baseProps} isOpen={false} />,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("renders form fields in add mode", () => {
    render(<MarketplaceModal {...baseProps} />);

    // Form fields are identified by their text content
    expect(screen.getByText("SETTINGS$MARKETPLACE_NAME")).toBeInTheDocument();
    expect(screen.getByText("SETTINGS$MARKETPLACE_SOURCE")).toBeInTheDocument();
    expect(screen.getByText("SETTINGS$MARKETPLACE_REF")).toBeInTheDocument();
    expect(
      screen.getByText("SETTINGS$MARKETPLACE_REPO_PATH"),
    ).toBeInTheDocument();
  });

  it("shows name required error when name is empty", async () => {
    render(<MarketplaceModal {...baseProps} />);
    const user = userEvent.setup();

    // Click save without entering anything
    await user.click(screen.getByTestId("marketplace-save-button"));

    expect(
      screen.getByText("SETTINGS$MARKETPLACE_NAME_REQUIRED"),
    ).toBeInTheDocument();
    expect(onSave).not.toHaveBeenCalled();
  });

  it("shows name invalid error for invalid name format", async () => {
    render(<MarketplaceModal {...baseProps} />);
    const user = userEvent.setup();

    // Find the name input by placeholder
    const nameInput = screen.getByPlaceholderText("e.g., my-skills");
    await user.type(nameInput, "123-invalid");
    await user.click(screen.getByTestId("marketplace-save-button"));

    expect(
      screen.getByText("SETTINGS$MARKETPLACE_NAME_INVALID"),
    ).toBeInTheDocument();
    expect(onSave).not.toHaveBeenCalled();
  });

  it("validates that onSave is not called when source is empty", async () => {
    render(<MarketplaceModal {...baseProps} />);
    const user = userEvent.setup();

    // Fill in name but not source
    const nameInput = screen.getByPlaceholderText("e.g., my-skills");
    await user.type(nameInput, "valid-name");
    await user.click(screen.getByTestId("marketplace-save-button"));

    // onSave should NOT have been called due to validation
    expect(onSave).not.toHaveBeenCalled();
  });

  it("clears name error when user starts typing", async () => {
    render(<MarketplaceModal {...baseProps} />);
    const user = userEvent.setup();

    // Trigger name required error
    await user.click(screen.getByTestId("marketplace-save-button"));
    expect(
      screen.getByText("SETTINGS$MARKETPLACE_NAME_REQUIRED"),
    ).toBeInTheDocument();

    // Start typing in name field
    const nameInput = screen.getByPlaceholderText("e.g., my-skills");
    await user.type(nameInput, "a");

    // Error should be cleared
    expect(
      screen.queryByText("SETTINGS$MARKETPLACE_NAME_REQUIRED"),
    ).not.toBeInTheDocument();
  });

  it("validates source before calling onSave", async () => {
    render(<MarketplaceModal {...baseProps} />);
    const user = userEvent.setup();

    // Fill name but not source
    const nameInput = screen.getByPlaceholderText("e.g., my-skills");
    await user.type(nameInput, "valid-name");
    await user.click(screen.getByTestId("marketplace-save-button"));

    // onSave should not be called
    expect(onSave).not.toHaveBeenCalled();

    // After filling source, onSave should be called
    const sourceInput = screen.getByPlaceholderText("github:owner/repo");
    await user.type(sourceInput, "github:owner/repo");
    await user.click(screen.getByTestId("marketplace-save-button"));

    expect(onSave).toHaveBeenCalledTimes(1);
  });

  it("calls onSave with correct data when form is valid", async () => {
    render(<MarketplaceModal {...baseProps} />);
    const user = userEvent.setup();

    await user.type(
      screen.getByPlaceholderText("e.g., my-skills"),
      "my-marketplace",
    );
    await user.type(
      screen.getByPlaceholderText("github:owner/repo"),
      "github:owner/repo",
    );
    await user.click(screen.getByTestId("marketplace-save-button"));

    expect(onSave).toHaveBeenCalledWith({
      name: "my-marketplace",
      source: "github:owner/repo",
      ref: undefined,
      repo_path: undefined,
      auto_load: false,
    });
  });

  it("rejects a URL entered in the repository path field", async () => {
    render(<MarketplaceModal {...baseProps} />);
    const user = userEvent.setup();

    await user.type(
      screen.getByPlaceholderText("e.g., my-skills"),
      "oh-extensions",
    );
    await user.type(
      screen.getByPlaceholderText("github:owner/repo"),
      "github:OpenHands/extensions",
    );
    await user.type(
      screen.getByPlaceholderText("e.g., marketplaces/internal"),
      "https://github.com/OpenHands/extensions",
    );
    await user.click(screen.getByTestId("marketplace-save-button"));

    expect(
      screen.getByText("SETTINGS$MARKETPLACE_REPO_PATH_INVALID"),
    ).toBeInTheDocument();
    expect(onSave).not.toHaveBeenCalled();
  });

  it("accepts a relative subdirectory as the repository path", async () => {
    render(<MarketplaceModal {...baseProps} />);
    const user = userEvent.setup();

    await user.type(screen.getByPlaceholderText("e.g., my-skills"), "team");
    await user.type(
      screen.getByPlaceholderText("github:owner/repo"),
      "github:acme/monorepo",
    );
    await user.type(
      screen.getByPlaceholderText("e.g., marketplaces/internal"),
      "marketplaces/internal",
    );
    await user.click(screen.getByTestId("marketplace-save-button"));

    expect(onSave).toHaveBeenCalledWith(
      expect.objectContaining({ repo_path: "marketplaces/internal" }),
    );
  });

  it("does not render scope selector unless scope selection is allowed", () => {
    render(<MarketplaceModal {...baseProps} />);

    expect(
      screen.queryByTestId("marketplace-scope-select"),
    ).not.toBeInTheDocument();
  });

  it("renders scope selector when scope selection is allowed (add mode)", () => {
    render(<MarketplaceModal {...baseProps} allowScopeSelection />);

    expect(screen.getByTestId("marketplace-scope-select")).toBeInTheDocument();
  });

  it("includes the chosen scope in onSave when scope selection is allowed", async () => {
    render(<MarketplaceModal {...baseProps} allowScopeSelection />);
    const user = userEvent.setup();

    await user.type(screen.getByPlaceholderText("e.g., my-skills"), "team");
    await user.type(
      screen.getByPlaceholderText("github:owner/repo"),
      "github:owner/repo",
    );
    await user.selectOptions(
      screen.getByTestId("marketplace-scope-select"),
      "org",
    );
    await user.click(screen.getByTestId("marketplace-save-button"));

    expect(onSave).toHaveBeenCalledWith(
      expect.objectContaining({ name: "team", scope: "org" }),
    );
  });

  it("calls onClose when cancel is clicked", async () => {
    render(<MarketplaceModal {...baseProps} />);

    await userEvent.click(
      screen.getByRole("button", { name: "BUTTON$CANCEL" }),
    );

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("disables buttons when isSaving is true", () => {
    render(<MarketplaceModal {...baseProps} isSaving={true} />);

    expect(screen.getByTestId("marketplace-save-button")).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "BUTTON$CANCEL" }),
    ).toBeDisabled();
  });

  it("renders in edit mode with read-only source", () => {
    render(
      <MarketplaceModal
        {...baseProps}
        mode="edit"
        marketplace={{
          name: "test-marketplace",
          source: "github:owner/repo",
          scope: "personal",
        }}
      />,
    );

    // Source field should be disabled in edit mode
    const sourceInput = screen.getByDisplayValue("github:owner/repo");
    expect(sourceInput).toBeDisabled();
    expect(
      screen.getByText("SETTINGS$MARKETPLACE_SOURCE_READONLY"),
    ).toBeInTheDocument();
  });
});
