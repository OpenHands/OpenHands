import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import type { LlmProfileSummary } from "#/api/settings-service/profiles-service.api";
import { DeleteProfileModal } from "#/components/features/settings/delete-profile-modal";

const deleteMock = vi.fn();
vi.mock("#/hooks/mutation/use-delete-llm-profile", () => ({
  useDeleteLlmProfile: () => ({ mutateAsync: deleteMock, isPending: false }),
}));

const toastMocks = vi.hoisted(() => ({
  displayErrorToast: vi.fn(),
  displaySuccessToast: vi.fn(),
}));
vi.mock("#/utils/custom-toast-handlers", () => toastMocks);

const profile: LlmProfileSummary = {
  name: "openai_gpt-4o",
  model: "openai/gpt-4o",
  base_url: null,
  api_key_set: true,
};

beforeEach(() => {
  deleteMock.mockReset().mockResolvedValue(undefined);
  toastMocks.displayErrorToast.mockReset();
  toastMocks.displaySuccessToast.mockReset();
});

describe("DeleteProfileModal", () => {
  it("renders nothing when profile is null", () => {
    const { container } = render(
      <DeleteProfileModal profile={null} onClose={vi.fn()} />,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("renders the confirmation copy interpolated with the profile name", () => {
    render(<DeleteProfileModal profile={profile} onClose={vi.fn()} />);

    expect(
      screen.getByText("SETTINGS$PROFILE_DELETE_CONFIRMATION"),
    ).toBeInTheDocument();
  });

  it("calls deleteProfile and onClose when the user confirms", async () => {
    const onClose = vi.fn();
    render(<DeleteProfileModal profile={profile} onClose={onClose} />);

    await userEvent.click(screen.getByTestId("delete-profile-confirm"));

    expect(deleteMock).toHaveBeenCalledWith("openai_gpt-4o");
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("calls onClose without invoking the mutation on Cancel", async () => {
    const onClose = vi.fn();
    render(<DeleteProfileModal profile={profile} onClose={onClose} />);

    await userEvent.click(
      screen.getByRole("button", { name: "BUTTON$CANCEL" }),
    );

    expect(deleteMock).not.toHaveBeenCalled();
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
