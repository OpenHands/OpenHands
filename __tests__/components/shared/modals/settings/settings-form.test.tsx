import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import SettingsService from "#/api/settings-service/settings-service.api";
import { SettingsForm } from "#/components/shared/modals/settings/settings-form";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { getAgentSettingValue } from "#/utils/sdk-settings-schema";

const trackSettingsSavedMock = vi.fn();
vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackSettingsSaved: trackSettingsSavedMock,
  }),
}));

describe("SettingsForm", () => {
  const onCloseMock = vi.fn();
  const saveSettingsSpy = vi.spyOn(SettingsService, "saveSettings");

  // The persisted llm.model is "openhands/<m>"; ModelSelector splits it into
  // provider ("openhands" → "OpenHands") and model name ("<m>") and feeds
  // them into two autocompletes. extractSettings reconstructs llm.model
  // from those two FormData entries on submit, so we wait for the model
  // autocomplete to be populated before triggering submission — otherwise
  // the spy would be called with model = undefined.
  const expectedModel = getAgentSettingValue(
    DEFAULT_SETTINGS,
    "llm.model",
  ) as string;
  const expectedModelName = expectedModel.split("/").slice(1).join("/");

  beforeEach(() => {
    vi.clearAllMocks();
    saveSettingsSpy.mockResolvedValue(true);
  });

  it("should save the user settings and close the modal when submitted outside a conversation route", async () => {
    renderWithProviders(
      <SettingsForm settings={DEFAULT_SETTINGS} onClose={onCloseMock} />,
      {
        navigation: { currentPath: "/settings" },
      },
    );

    await waitFor(() => {
      expect(screen.getByTestId("llm-model-input")).toHaveValue(
        expectedModelName,
      );
    });

    // The Autocomplete popover may be open at this point; click on the save
    // button can be consumed by the outside-click handler. Submit the form
    // element directly — the production onSubmit handler is what we're
    // exercising here, not the click semantics.
    fireEvent.submit(screen.getByTestId("settings-form"));

    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          agent_settings_diff: expect.objectContaining({
            llm: expect.objectContaining({
              model: expectedModel,
            }),
          }),
        }),
      );
    });
    await waitFor(() => {
      expect(onCloseMock).toHaveBeenCalled();
    });
  });

  it("should confirm before saving when submitted from a conversation route", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <SettingsForm settings={DEFAULT_SETTINGS} onClose={onCloseMock} />,
      {
        navigation: {
          currentPath: "/conversations/test-conversation-id",
          conversationId: "test-conversation-id",
        },
      },
    );

    await waitFor(() => {
      expect(screen.getByTestId("llm-model-input")).toHaveValue(
        expectedModelName,
      );
    });

    fireEvent.submit(screen.getByTestId("settings-form"));

    expect(saveSettingsSpy).not.toHaveBeenCalled();

    // The confirm modal mounts after the form submit handler runs the
    // conversation-route branch, so wait for it instead of using a sync
    // getByRole.
    const confirmButton = await screen.findByRole("button", {
      name: /BUTTON\$END_SESSION|end session/i,
    });

    await user.click(confirmButton);

    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          agent_settings_diff: expect.objectContaining({
            llm: expect.objectContaining({
              model: expectedModel,
            }),
          }),
        }),
      );
    });
    await waitFor(() => {
      expect(onCloseMock).toHaveBeenCalled();
    });
  });

  it("calls trackSettingsSaved with LLM details when form is submitted", async () => {
    renderWithProviders(
      <SettingsForm settings={DEFAULT_SETTINGS} onClose={onCloseMock} />,
      {
        navigation: { currentPath: "/settings" },
      },
    );

    await waitFor(() => {
      expect(screen.getByTestId("llm-model-input")).toHaveValue(
        expectedModelName,
      );
    });

    fireEvent.submit(screen.getByTestId("settings-form"));

    await waitFor(() => {
      expect(trackSettingsSavedMock).toHaveBeenCalledWith(
        expect.objectContaining({
          llmApiKeySet: expect.stringMatching(/^(SET|UNSET)$/),
          searchApiKeySet: expect.stringMatching(/^(SET|UNSET)$/),
        }),
      );
    });
  });
});
