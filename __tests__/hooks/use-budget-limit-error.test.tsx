import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { createInstance } from "i18next";
import { I18nextProvider } from "react-i18next";
import {
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import type { CloudMyBudget } from "#/api/cloud/types";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { useBudgetLimitError } from "#/hooks/use-budget-limit-error";
import { translationResources } from "#/i18n/resources";

vi.unmock("react-i18next");

const getCloudMyBudgetMock = vi.fn();

vi.mock("#/api/cloud/organization-service.api", () => ({
  getCloudMyBudget: (...args: unknown[]) => getCloudMyBudgetMock(...args),
}));

const USER_BUDGET_ERROR =
  "litellm.RateLimitError: RateLimitError: Litellm_proxyException - Budget has been exceeded! TeamMember=user-1:org-1 Current cost: 348.96549289999996, Max budget: 348.9626492500001";
const USER_BUDGET_ERROR_OLDER_WORDING =
  "litellm.RateLimitError: Budget has been exceeded! User=user-1 in Team=org-1 Current cost: 348.97, Max budget: 348.96";
const ORG_BUDGET_ERROR =
  "litellm.RateLimitError: Budget has been exceeded! Team=org-1 Current cost: 2913.4, Max budget: 2913.1";

const cloudBackend: Backend = {
  id: "ohe",
  name: "OpenHands Enterprise",
  host: "https://ohe.example.com",
  apiKey: "bearer-token",
  kind: "cloud",
};

const budget: CloudMyBudget = {
  enabled: true,
  monthly_limit: 500,
  org_monthly_limit: 2000.5,
};

const i18n = createInstance();

beforeAll(async () => {
  await i18n.init({
    lng: "en",
    resources: { en: { openhands: translationResources.en } },
    ns: ["openhands"],
    defaultNS: "openhands",
    interpolation: { escapeValue: false },
  });
});

function renderBudgetLimitError(errorMessage: string | null) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <I18nextProvider i18n={i18n}>
          <ActiveBackendProvider>{children}</ActiveBackendProvider>
        </I18nextProvider>
      </QueryClientProvider>
    );
  }
  const { result } = renderHook(() => useBudgetLimitError(errorMessage), {
    wrapper: Wrapper,
  });
  return { result, queryClient };
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([cloudBackend]);
  setActiveSelection({ backendId: cloudBackend.id, orgId: "org-1" });
  getCloudMyBudgetMock.mockReset();
  getCloudMyBudgetMock.mockResolvedValue(budget);
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("useBudgetLimitError", () => {
  it.each([USER_BUDGET_ERROR, USER_BUDGET_ERROR_OLDER_WORDING])(
    "names the user budget and its limit when the member's own budget is reached",
    async (errorMessage) => {
      // Act
      const { result } = renderBudgetLimitError(errorMessage);

      // Assert
      await waitFor(() =>
        expect(result.current?.message).toBe(
          "You've reached your monthly user budget of $500.",
        ),
      );
    },
  );

  it("names the organization budget and its limit when the organization's budget is reached", async () => {
    // Act
    const { result } = renderBudgetLimitError(ORG_BUDGET_ERROR);

    // Assert
    await waitFor(() =>
      expect(result.current?.message).toBe(
        "Your organization has reached its monthly budget of $2,000.50.",
      ),
    );
  });

  it("adds the admin's instructions below the summary", async () => {
    // Arrange
    getCloudMyBudgetMock.mockResolvedValue({
      ...budget,
      limit_reached_instructions: "Contact #it-help to request an increase.",
    });

    // Act
    const { result } = renderBudgetLimitError(USER_BUDGET_ERROR);

    // Assert
    await waitFor(() =>
      expect(result.current?.message).toBe(
        "You've reached your monthly user budget of $500.\nContact #it-help to request an increase.",
      ),
    );
  });

  it("links to the Your Budget page of the active organization", async () => {
    // Act
    const { result } = renderBudgetLimitError(USER_BUDGET_ERROR);

    // Assert
    await waitFor(() =>
      expect(result.current?.link).toEqual({
        href: "https://ohe.example.com/settings/your-budget?org=org-1",
        label: "View your budget",
        openInNewTab: true,
      }),
    );
  });

  it.each([
    [
      "budgets are off for the organization",
      () => getCloudMyBudgetMock.mockResolvedValue({ enabled: false }),
    ],
    [
      "the budget that was reached has no limit",
      () =>
        getCloudMyBudgetMock.mockResolvedValue({
          ...budget,
          monthly_limit: null,
        }),
    ],
    [
      "the budget cannot be read",
      () => getCloudMyBudgetMock.mockRejectedValue(new Error("Forbidden")),
    ],
  ])("keeps the raw error when %s", async (_case, arrangeBudget) => {
    // Arrange
    arrangeBudget();

    // Act
    const { result, queryClient } = renderBudgetLimitError(USER_BUDGET_ERROR);
    await waitFor(() => expect(getCloudMyBudgetMock).toHaveBeenCalled());
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));

    // Assert
    expect(result.current).toBeNull();
  });

  it("leaves other errors alone without reading the budget", () => {
    // Act
    const { result } = renderBudgetLimitError(
      "litellm.RateLimitError: Rate limit reached, retry in 20s",
    );

    // Assert
    expect(result.current).toBeNull();
    expect(getCloudMyBudgetMock).not.toHaveBeenCalled();
  });

  it("does not read the budget on a local backend", () => {
    // Arrange
    setRegisteredBackends([
      {
        id: "local",
        name: "Local",
        host: "http://localhost:8000",
        apiKey: "session-key",
        kind: "local",
      },
    ]);

    // Act
    const { result } = renderBudgetLimitError(USER_BUDGET_ERROR);

    // Assert
    expect(result.current).toBeNull();
    expect(getCloudMyBudgetMock).not.toHaveBeenCalled();
  });
});
