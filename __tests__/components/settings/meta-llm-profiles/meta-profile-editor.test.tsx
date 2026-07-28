import { describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { MetaProfileEditor } from "#/components/features/settings/meta-llm-profiles";
import type { MetaProfile } from "#/api/meta-profiles-service/meta-profiles-service.api";

const AVAILABLE = ["minimax", "gpt", "deepseek"];

const FILLED: MetaProfile = {
  classifier_model: "minimax",
  default_model: "gpt",
  classes: [],
  prompt_template:
    "Return JSON with the best model.\n{{ model_table }}\nTask:\n{{ instance_text }}",
  model_table: "- GPT-5.4\n- MiniMax-M3",
  target_models: [{ model: "GPT-5.4", profile: "deepseek" }],
};

describe("MetaProfileEditor", () => {
  it("disables Save in create mode until required fields are set", () => {
    renderWithProviders(
      <MetaProfileEditor
        mode="create"
        availableProfiles={AVAILABLE}
        isSaving={false}
        onSave={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByTestId("meta-profile-save")).toBeDisabled();
  });

  it("enables Save in edit mode with a complete config and saves it", async () => {
    const user = userEvent.setup();
    const onSave = vi.fn();
    renderWithProviders(
      <MetaProfileEditor
        mode="edit"
        initialName="balanced"
        initialConfig={FILLED}
        availableProfiles={AVAILABLE}
        isSaving={false}
        onSave={onSave}
        onCancel={vi.fn()}
      />,
    );

    const save = screen.getByTestId("meta-profile-save");
    expect(save).toBeEnabled();

    await user.click(save);

    expect(onSave).toHaveBeenCalledWith("balanced", FILLED);
  });

  it("disables the name field in edit mode", () => {
    renderWithProviders(
      <MetaProfileEditor
        mode="edit"
        initialName="balanced"
        initialConfig={FILLED}
        availableProfiles={AVAILABLE}
        isSaving={false}
        onSave={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByTestId("meta-profile-name-input")).toBeDisabled();
  });

  it("adds and removes target model rows", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <MetaProfileEditor
        mode="create"
        availableProfiles={AVAILABLE}
        isSaving={false}
        onSave={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(
      screen.getByTestId("meta-profile-target-models-empty"),
    ).toBeInTheDocument();

    await user.click(screen.getByTestId("meta-profile-add-target-model"));
    expect(
      screen.getByTestId("meta-profile-target-model-label-0"),
    ).toBeInTheDocument();

    await user.click(screen.getByTestId("meta-profile-remove-target-model-0"));
    expect(
      screen.queryByTestId("meta-profile-target-model-label-0"),
    ).not.toBeInTheDocument();
  });

  it("requires the prompt template to include the instance_text placeholder", () => {
    renderWithProviders(
      <MetaProfileEditor
        mode="edit"
        initialName="balanced"
        initialConfig={{
          ...FILLED,
          prompt_template: "Return JSON with the best model.",
        }}
        availableProfiles={AVAILABLE}
        isSaving={false}
        onSave={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByTestId("meta-profile-save")).toBeDisabled();
  });

  it("rejects a duplicate name in create mode and blocks Save", async () => {
    const user = userEvent.setup();
    const onSave = vi.fn();
    renderWithProviders(
      <MetaProfileEditor
        mode="create"
        initialConfig={FILLED}
        availableProfiles={AVAILABLE}
        existingNames={["balanced"]}
        isSaving={false}
        onSave={onSave}
        onCancel={vi.fn()}
      />,
    );

    await user.type(screen.getByTestId("meta-profile-name-input"), "balanced");

    expect(screen.getByTestId("meta-profile-name-taken")).toBeInTheDocument();
    const save = screen.getByTestId("meta-profile-save");
    expect(save).toBeDisabled();

    await user.click(save);
    expect(onSave).not.toHaveBeenCalled();
  });

  it("accepts a unique name in create mode", async () => {
    const user = userEvent.setup();
    const onSave = vi.fn();
    renderWithProviders(
      <MetaProfileEditor
        mode="create"
        initialConfig={FILLED}
        availableProfiles={AVAILABLE}
        existingNames={["balanced"]}
        isSaving={false}
        onSave={onSave}
        onCancel={vi.fn()}
      />,
    );

    await user.type(screen.getByTestId("meta-profile-name-input"), "fast");

    expect(
      screen.queryByTestId("meta-profile-name-taken"),
    ).not.toBeInTheDocument();
    const save = screen.getByTestId("meta-profile-save");
    expect(save).toBeEnabled();

    await user.click(save);
    expect(onSave).toHaveBeenCalledWith("fast", FILLED);
  });

  it("allows the existing name in edit mode (no duplicate warning)", () => {
    renderWithProviders(
      <MetaProfileEditor
        mode="edit"
        initialName="balanced"
        initialConfig={FILLED}
        availableProfiles={AVAILABLE}
        existingNames={["balanced"]}
        isSaving={false}
        onSave={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(
      screen.queryByTestId("meta-profile-name-taken"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("meta-profile-save")).toBeEnabled();
  });

  it("calls onCancel when Cancel is clicked", async () => {
    const user = userEvent.setup();
    const onCancel = vi.fn();
    renderWithProviders(
      <MetaProfileEditor
        mode="create"
        availableProfiles={AVAILABLE}
        isSaving={false}
        onSave={vi.fn()}
        onCancel={onCancel}
      />,
    );

    await user.click(screen.getByTestId("meta-profile-cancel"));
    expect(onCancel).toHaveBeenCalled();
  });
});
