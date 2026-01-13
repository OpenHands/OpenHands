import React, { PropsWithChildren } from "react";
import { RenderOptions, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { I18nextProvider, initReactI18next } from "react-i18next";
import i18n from "i18next";
import { expect, vi } from "vitest";
import { AxiosError } from "axios";
import userEvent from "@testing-library/user-event";
import { INITIAL_MOCK_ORGS } from "#/mocks/org-handlers";

export const useParamsMock = vi.fn(() => ({
  conversationId: "test-conversation-id",
}));

// Mock useParams before importing components
vi.mock("react-router", async () => {
  const actual =
    await vi.importActual<typeof import("react-router")>("react-router");
  return {
    ...actual,
    useParams: useParamsMock,
    useRevalidator: () => ({
      revalidate: vi.fn(),
    }),
  };
});

// Initialize i18n for tests
i18n.use(initReactI18next).init({
  lng: "en",
  fallbackLng: "en",
  ns: ["translation"],
  defaultNS: "translation",
  resources: {
    en: {
      translation: {},
    },
  },
  interpolation: {
    escapeValue: false,
  },
});

// This type interface extends the default options for render from RTL
interface ExtendedRenderOptions extends Omit<RenderOptions, "queries"> {}

// Export our own customized renderWithProviders function that renders with QueryClient and i18next providers
// Since we're using Zustand stores, we don't need a Redux Provider wrapper
export function renderWithProviders(
  ui: React.ReactElement,
  renderOptions: ExtendedRenderOptions = {},
) {
  function Wrapper({ children }: PropsWithChildren) {
    return (
      <QueryClientProvider
        client={
          new QueryClient({
            defaultOptions: { queries: { retry: false } },
          })
        }
      >
        <I18nextProvider i18n={i18n}>{children}</I18nextProvider>
      </QueryClientProvider>
    );
  }
  return render(ui, { wrapper: Wrapper, ...renderOptions });
}

export const createAxiosNotFoundErrorObject = () =>
  new AxiosError(
    "Request failed with status code 404",
    "ERR_BAD_REQUEST",
    undefined,
    undefined,
    {
      status: 404,
      statusText: "Not Found",
      data: { message: "Settings not found" },
      headers: {},
      // @ts-expect-error - we only need the response object for this test
      config: {},
    },
  );

export const selectOrganization = async ({
  orgIndex,
}: {
  orgIndex: number;
}) => {
  const organizationSelect = await screen.findByTestId("org-select");
  expect(organizationSelect).toBeInTheDocument();

  await userEvent.click(organizationSelect);

  // Wait for the options to appear in the popover
  const targetOrg = INITIAL_MOCK_ORGS[orgIndex];
  if (!targetOrg) {
    expect.fail(`No organization found at index ${orgIndex}`);
  }

  // Find the option by its text content (organization name)
  const option = await screen.findByText(targetOrg.name);
  await userEvent.click(option);
};

export const createAxiosError = (
  status: number,
  statusText: string,
  data: unknown,
) =>
  new AxiosError(
    `Request failed with status code ${status}`,
    "ERR_BAD_REQUEST",
    undefined,
    undefined,
    {
      status,
      statusText,
      data,
      headers: {},
      // @ts-expect-error - we only need the response object for this test
      config: {},
    },
  );
