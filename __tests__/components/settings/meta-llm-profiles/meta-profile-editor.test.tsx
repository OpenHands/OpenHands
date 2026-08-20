import { describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { MetaProfileEditor } from "#/components/features/settings/meta-llm-profiles";
import type { MetaProfile } from "#/api/meta-profiles-service/meta-profiles-service.api";
import {
  DEFAULT_MAX_SCORE_PARETO_META_PROFILE_DEFAULT,
  DEFAULT_MAX_SCORE_PARETO_META_PROFILE_NAME,
} from "#/components/features/settings/meta-llm-profiles/default-meta-profile";

const AVAILABLE = ["minimax", "minimax-m3", "gpt", "deepseek"];

const CONNECTIONS = [
  {
    id: "conn-openhands",
    display_name: "OpenHands",
    provider: "openhands",
    base_url: null,
    created_at: 0,
    updated_at: 0,
    api_key_set: true,
  },
];

const FILLED: MetaProfile = {
  classifier_model: "minimax",
  default_model: "gpt",
  classes: [],
  prompt_template:
    "Return JSON with the best model.\n{{ model_table }}\nTask:\n{{ instance_text }}",
  model_table: "- GPT-5.4\n- MiniMax-M3",
};

describe("MetaProfileEditor", () => {
  it("prefills create mode with the max-score Pareto router", async () => {
    const user = userEvent.setup();
    const onSave = vi.fn();
    renderWithProviders(
      <MetaProfileEditor
        mode="create"
        providerConnections={CONNECTIONS}
        selectRouterConnectionByDefault
        availableProfiles={AVAILABLE}
        isSaving={false}
        onSave={onSave}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByTestId("meta-profile-name-input")).toHaveValue(
      DEFAULT_MAX_SCORE_PARETO_META_PROFILE_NAME,
    );
    expect(screen.getByTestId("meta-profile-classifier-input")).toHaveValue(
      "minimax-m3",
    );
    expect(screen.getByTestId("meta-profile-default-input")).toHaveValue(
      "minimax-m3",
    );
    expect(screen.getByTestId("meta-profile-prompt-template")).toHaveValue(
      DEFAULT_MAX_SCORE_PARETO_META_PROFILE_DEFAULT.prompt_template,
    );
    expect(screen.getByTestId("meta-profile-model-table")).toHaveValue(
      DEFAULT_MAX_SCORE_PARETO_META_PROFILE_DEFAULT.model_table,
    );
    expect(screen.getByTestId("meta-profile-router-connection")).toHaveValue(
      "OpenHands (openhands)",
    );

    await user.click(screen.getByTestId("meta-profile-save"));

    expect(onSave).toHaveBeenCalledWith(
      DEFAULT_MAX_SCORE_PARETO_META_PROFILE_NAME,
      DEFAULT_MAX_SCORE_PARETO_META_PROFILE_DEFAULT,
      "conn-openhands",
    );
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

    expect(onSave).toHaveBeenCalledWith("balanced", FILLED, null);
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

    await user.clear(screen.getByTestId("meta-profile-name-input"));
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
        providerConnections={CONNECTIONS}
        selectRouterConnectionByDefault
        availableProfiles={AVAILABLE}
        existingNames={["balanced"]}
        isSaving={false}
        onSave={onSave}
        onCancel={vi.fn()}
      />,
    );

    await user.clear(screen.getByTestId("meta-profile-name-input"));
    await user.type(screen.getByTestId("meta-profile-name-input"), "fast");

    expect(
      screen.queryByTestId("meta-profile-name-taken"),
    ).not.toBeInTheDocument();
    const save = screen.getByTestId("meta-profile-save");
    expect(save).toBeEnabled();

    await user.click(save);
    expect(onSave).toHaveBeenCalledWith("fast", FILLED, "conn-openhands");
  });

  it("returns no connection when the picker is set to don't create", async () => {
    const user = userEvent.setup();
    const onSave = vi.fn();
    renderWithProviders(
      <MetaProfileEditor
        mode="create"
        initialConfig={FILLED}
        providerConnections={CONNECTIONS}
        availableProfiles={AVAILABLE}
        isSaving={false}
        onSave={onSave}
        onCancel={vi.fn()}
      />,
    );

    await user.clear(screen.getByTestId("meta-profile-name-input"));
    await user.type(screen.getByTestId("meta-profile-name-input"), "fast");
    // Without selectRouterConnectionByDefault the picker starts on "don't
    // create", so no provider connection is returned.
    await user.click(screen.getByTestId("meta-profile-save"));

    expect(onSave).toHaveBeenCalledWith("fast", FILLED, null);
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
