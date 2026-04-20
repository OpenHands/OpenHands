import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  MOCK_DEFAULT_USER_SETTINGS,
  resetTestHandlersMockSettings,
} from "#/mocks/handlers";
import MCPSettingsScreen, { clientLoader } from "#/routes/mcp-settings";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";
import { Settings } from "#/types/settings";

function buildSettings(overrides: Partial<Settings> = {}): Settings {
  return {
    ...MOCK_DEFAULT_USER_SETTINGS,
    ...overrides,
    agent_settings: {
      ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
      ...overrides.agent_settings,
    },
  };
}

function deepMerge(
  base: Record<string, unknown>,
  patch: Record<string, unknown>,
): Record<string, unknown> {
  const result = { ...base };

  for (const [key, value] of Object.entries(patch)) {
    if (
      value != null &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      result[key] != null &&
      typeof result[key] === "object" &&
      !Array.isArray(result[key])
    ) {
      result[key] = deepMerge(
        result[key] as Record<string, unknown>,
        value as Record<string, unknown>,
      );
    } else {
      result[key] = value;
    }
  }

  return result;
}

function renderMcpSettingsScreen() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  useSelectedOrganizationStore.setState({ organizationId: "1" });

  return render(<MCPSettingsScreen />, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });
}

beforeEach(() => {
  vi.restoreAllMocks();
  resetTestHandlersMockSettings();
  useSelectedOrganizationStore.setState({ organizationId: "1" });
});

describe("MCPSettingsScreen", () => {
  it("renders and saves the built-in Tavily search settings on the MCP page", async () => {
    let persistedSettings = buildSettings({
      search_api_key: "tavily-key",
      search_api_key_set: true,
    });

    vi.spyOn(SettingsService, "getSettings").mockImplementation(async () =>
      structuredClone(persistedSettings),
    );
    const saveSettingsSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockImplementation(async (payload) => {
        persistedSettings = buildSettings(
          deepMerge(
            structuredClone(persistedSettings) as Record<string, unknown>,
            payload as Record<string, unknown>,
          ) as Partial<Settings>,
        );
        return true;
      });

    renderMcpSettingsScreen();

    expect(
      await screen.findByTestId("mcp-search-settings-section"),
    ).toBeInTheDocument();
    const searchApiKeyInput = await screen.findByTestId("search-api-key-input");
    expect(searchApiKeyInput).toHaveValue("tavily-key");

    await userEvent.clear(searchApiKeyInput);
    await userEvent.type(searchApiKeyInput, "updated-tavily-key");
    await userEvent.click(screen.getByTestId("save-search-api-key-button"));

    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith({
        search_api_key: "updated-tavily-key",
      });
    });

    await waitFor(() => {
      expect(screen.getByTestId("search-api-key-input")).toHaveValue(
        "updated-tavily-key",
      );
    });
  });

  it("removes a newly added MCP server after the delete flow completes", async () => {
    let persistedSettings = buildSettings();

    vi.spyOn(SettingsService, "getSettings").mockImplementation(async () =>
      structuredClone(persistedSettings),
    );
    vi.spyOn(SettingsService, "saveSettings").mockImplementation(
      async (payload) => {
        persistedSettings = buildSettings(
          deepMerge(
            structuredClone(persistedSettings) as Record<string, unknown>,
            payload as Record<string, unknown>,
          ) as Partial<Settings>,
        );
        return true;
      },
    );

    renderMcpSettingsScreen();

    await screen.findByText("SETTINGS$MCP_NO_SERVERS");

    await userEvent.click(
      screen.getByRole("button", { name: "SETTINGS$MCP_ADD_SERVER" }),
    );
    await userEvent.type(
      await screen.findByTestId("url-input"),
      "https://mcp.example.com/sse",
    );
    await userEvent.click(screen.getByTestId("submit-button"));

    await waitFor(() => {
      expect(screen.getAllByTestId("mcp-server-item")).toHaveLength(1);
    });

    await userEvent.click(screen.getByTestId("delete-mcp-server-button"));
    await userEvent.click(await screen.findByTestId("confirm-button"));

    await waitFor(() => {
      expect(screen.queryAllByTestId("mcp-server-item")).toHaveLength(0);
    });
  });
});

describe("clientLoader permission checks", () => {
  it("should export a clientLoader for route protection", () => {
    expect(clientLoader).toBeDefined();
    expect(typeof clientLoader).toBe("function");
  });
});
