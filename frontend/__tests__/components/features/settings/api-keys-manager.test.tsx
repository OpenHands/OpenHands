import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ApiKeysManager } from "#/components/features/settings/api-keys-manager";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";

// Mock the react-i18next
vi.mock("react-i18next", async () => {
  const actual = await vi.importActual<typeof import("react-i18next")>("react-i18next");
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string) => key,
    }),
    Trans: ({ i18nKey, components }: { i18nKey: string; components: Record<string, React.ReactNode> }) => {
      // Simplified Trans component that renders the link
      if (i18nKey === "SETTINGS$API_KEYS_DESCRIPTION") {
        return (
          <span>
            API keys allow you to authenticate with the OpenHands API programmatically.
            Keep your API keys secure; anyone with your API key can access your account.
            For more information on how to use the API, see our {components.a}
          </span>
        );
      }
      return <span>{i18nKey}</span>;
    },
  };
});

// Mock the API keys hook
vi.mock("#/hooks/query/use-api-keys", () => ({
  useApiKeys: () => ({
    data: [],
    isLoading: false,
    error: null,
  }),
}));

describe("ApiKeysManager", () => {
  beforeEach(() => {
    useSelectedOrganizationStore.setState({ organizationId: "test-org-id" });
  });

  const renderComponent = () => {
    const queryClient = new QueryClient();
    return render(
      <QueryClientProvider client={queryClient}>
        <ApiKeysManager />
      </QueryClientProvider>
    );
  };

  it("should render the API documentation link", () => {
    renderComponent();

    // Find the link to the API documentation
    const link = screen.getByRole("link");
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute("href", "https://docs.all-hands.dev/usage/cloud/cloud-api");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });
});
