import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import type { LlmProfileSummary } from "#/api/settings-service/profiles-service.api";
import { RenameProfileModal } from "#/components/features/settings/rename-profile-modal";

const renameMock = vi.fn();
vi.mock("#/hooks/mutation/use-rename-llm-profile", () => ({
  useRenameLlmProfile: () => ({ mutateAsync: renameMock, isPending: false }),
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
  renameMock.mockReset().mockResolvedValue(undefined);
  toastMocks.displayErrorToast.mockReset();
  toastMocks.displaySuccessToast.mockReset();
});

describe("RenameProfileModal", () => {
  it("renders nothing when profile is null", () => {
    const { container } = render(
      <RenameProfileModal profile={null} onClose={vi.fn()} />,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("prefills the input with the current profile name", () => {
    render(<RenameProfileModal profile={profile} onClose={vi.fn()} />);

    const input = screen.getByTestId(
      "rename-profile-input",
    ) as HTMLInputElement;
    expect(input.value).toBe("openai_gpt-4o");
  });

  it("submits the trimmed new name and closes on success", async () => {
    const onClose = vi.fn();
    render(<RenameProfileModal profile={profile} onClose={onClose} />);
    const user = userEvent.setup();

    const input = screen.getByTestId("rename-profile-input");
    await user.clear(input);
    await user.type(input, "  new_name  ");
    await user.click(screen.getByTestId("rename-profile-submit"));

    expect(renameMock).toHaveBeenCalledWith({
      name: "openai_gpt-4o",
      newName: "new_name",
    });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("disables submit and skips the mutation when the input is invalid", async () => {
    render(<RenameProfileModal profile={profile} onClose={vi.fn()} />);
    const user = userEvent.setup();

    const input = screen.getByTestId("rename-profile-input");
    await user.clear(input);
    await user.type(input, "has space");

    expect(screen.getByTestId("rename-profile-submit")).toBeDisabled();
    expect(renameMock).not.toHaveBeenCalled();
  });

  it("closes without renaming when the name is unchanged", async () => {
    const onClose = vi.fn();
    render(<RenameProfileModal profile={profile} onClose={onClose} />);

    await userEvent.click(screen.getByTestId("rename-profile-submit"));

    expect(renameMock).not.toHaveBeenCalled();
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
