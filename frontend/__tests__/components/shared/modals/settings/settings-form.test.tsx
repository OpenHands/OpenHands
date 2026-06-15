import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { createRoutesStub } from "react-router";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import SettingsService from "#/api/settings-service/settings-service.api";
import { SettingsForm } from "#/components/shared/modals/settings/settings-form";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { getAgentSettingValue } from "#/utils/sdk-settings-schema";

describe("SettingsForm", () => {
  const onCloseMock = vi.fn();
  const saveSettingsSpy = vi.spyOn(SettingsService, "saveSettings");

  const RouteStub = createRoutesStub([
    {
      Component: () => (
        <SettingsForm settings={DEFAULT_SETTINGS} onClose={onCloseMock} />
      ),
      path: "/",
    },
  ]);

  beforeEach(() => {
    vi.clearAllMocks();
    saveSettingsSpy.mockResolvedValue(true);
  });

  it("should save the user settings and close the modal when the form is submitted", async () => {
    renderWithProviders(<RouteStub />);

    await waitFor(() =>
      expect(screen.getByTestId("llm-model-input")).toHaveValue(
        "claude-opus-4-5-20251101",
      ),
    );

    fireEvent.submit(screen.getByTestId("settings-form"));

    await waitFor(() =>
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          agent_settings_diff: expect.objectContaining({
            llm: expect.objectContaining({
              model: getAgentSettingValue(DEFAULT_SETTINGS, "llm.model"),
            }),
          }),
        }),
      ),
    );
    await waitFor(() => expect(onCloseMock).toHaveBeenCalled());
  });
});
