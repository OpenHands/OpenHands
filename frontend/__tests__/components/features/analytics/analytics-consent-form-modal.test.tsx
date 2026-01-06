import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AnalyticsConsentFormModal } from "#/components/features/analytics/analytics-consent-form-modal";
import SettingsService from "#/api/settings-service/settings-service.api";

describe("AnalyticsConsentFormModal", () => {
  it("should call saveUserSettings with consent", async () => {
    const user = userEvent.setup();
    const onCloseMock = vi.fn();
    const saveUserSettingsSpy = vi.spyOn(SettingsService, "saveSettings");

    render(<AnalyticsConsentFormModal onClose={onCloseMock} />, {
      wrapper: ({ children }) => (
        <QueryClientProvider client={new QueryClient()}>
          {children}
        </QueryClientProvider>
      ),
    });

    const confirmButton = screen.getByTestId("confirm-preferences");
    await user.click(confirmButton);

    expect(saveUserSettingsSpy).toHaveBeenCalledWith(
      expect.objectContaining({ user_consents_to_analytics: true }),
    );
    await waitFor(() => expect(onCloseMock).toHaveBeenCalled());
  });
});
