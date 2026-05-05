import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { createRoutesStub } from "react-router";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import SettingsService from "#/api/settings-service/settings-service.api";
import { SettingsForm } from "#/components/shared/modals/settings/settings-form";
import { DEFAULT_SETTINGS } from "#/services/settings";

describe("SettingsForm", () => {
  const onCloseMock = vi.fn();

  const RouteStub = createRoutesStub([
    {
      Component: () => (
        <SettingsForm settings={DEFAULT_SETTINGS} onClose={onCloseMock} />
      ),
      path: "/",
    },
  ]);

  it("should save the user settings and close the modal when the form is submitted", async () => {
    const saveSettingsSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);
    renderWithProviders(<RouteStub />);

    fireEvent.change(screen.getByTestId("llm-api-key-input"), {
      target: { value: "new-key" },
    });
    fireEvent.submit(screen.getByTestId("settings-form"));

    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          agent_settings_diff: expect.objectContaining({
            llm: expect.objectContaining({
              api_key: "new-key",
            }),
          }),
        }),
      );
    });
    await waitFor(() => expect(onCloseMock).toHaveBeenCalled());
  });
});
