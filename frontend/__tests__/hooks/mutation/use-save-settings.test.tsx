import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import SettingsService from "#/api/settings-service/settings-service.api";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";

describe("useSaveSettings", () => {
  beforeEach(() => {
    useSelectedOrganizationStore.setState({ organizationId: "test-org-id" });
  });

  it("should preserve canonical llm.api_key payload values", async () => {
    const saveSettingsSpy = vi.spyOn(SettingsService, "saveSettings");
    const { result } = renderHook(() => useSaveSettings(), {
      wrapper: ({ children }) => (
        <QueryClientProvider client={new QueryClient()}>
          {children}
        </QueryClientProvider>
      ),
    });

    result.current.mutate({ "llm.api_key": "" });
    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          "llm.api_key": "",
        }),
      );
    });

    result.current.mutate({ "llm.api_key": null });
    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          "llm.api_key": null,
        }),
      );
    });
  });
});
