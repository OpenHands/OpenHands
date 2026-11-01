import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import {
  getCloudOrganizationMe,
  getCloudOrganizations,
} from "#/api/cloud/organization-service.api";
import { AutomationListRow } from "#/components/features/automations/automation-list-row";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";

// `automation-list-row.test.tsx` mocks the permission hooks wholesale. These
// tests run the real hooks against a mocked `/me` service so the
// creator-vs-manager rule for the enabled toggle is exercised end to end.
vi.mock("#/api/cloud/organization-service.api", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("#/api/cloud/organization-service.api")
  >()),
  getCloudOrganizations: vi.fn(),
  getCloudOrganizationMe: vi.fn(),
}));

vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => ({ navigate: vi.fn(), currentPath: "/" }),
}));

const ORG_ID = "org-1";

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-key",
  kind: "cloud",
};

function makeAutomation(overrides: Partial<Automation> = {}): Automation {
  return {
    id: "auto-1",
    name: "Teammate digest",
    prompt: "Summarize",
    trigger: {
      type: "cron",
      schedule: "0 9 * * *",
      schedule_human: "Daily at 09:00",
    },
    enabled: true,
    user_id: "creator-user",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

function renderRow(automation: Automation) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>
        <AutomationListRow
          automation={automation}
          onToggle={vi.fn()}
          onRunNow={vi.fn()}
          onExport={vi.fn()}
          onDelete={vi.fn()}
        />
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

async function openActionsMenu(user: ReturnType<typeof userEvent.setup>) {
  // Run now only renders once the caller's permissions have resolved.
  await screen.findByTestId("automation-run-now-auto-1");
  await user.click(screen.getByLabelText(I18nKey.AUTOMATIONS$ACTIONS_MENU));
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([cloudBackend]);
  setActiveSelection({ backendId: cloudBackend.id, orgId: ORG_ID });
  vi.mocked(getCloudOrganizations).mockResolvedValue({
    items: [],
    currentOrgId: null,
  });
  // A manager (has manage_automations) who did not create the automation.
  vi.mocked(getCloudOrganizationMe).mockResolvedValue({
    orgId: ORG_ID,
    userId: "manager-user",
    role: "admin",
    permissions: ["view_automations", "manage_automations"],
  });
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("AutomationListRow — non-creator manager", () => {
  it("offers Turn off and Delete on an enabled automation", async () => {
    // Arrange
    const user = userEvent.setup();
    renderRow(makeAutomation({ enabled: true }));

    // Act
    await openActionsMenu(user);

    // Assert
    expect(
      screen.getByRole("button", { name: I18nKey.AUTOMATIONS$TURN_OFF }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: I18nKey.AUTOMATIONS$DELETE }),
    ).toBeInTheDocument();
  });

  it("hides Turn on but keeps Delete on a disabled automation", async () => {
    // Arrange
    const user = userEvent.setup();
    renderRow(makeAutomation({ enabled: false }));

    // Act
    await openActionsMenu(user);

    // Assert
    expect(
      screen.queryByRole("button", { name: I18nKey.AUTOMATIONS$TURN_ON }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: I18nKey.AUTOMATIONS$DELETE }),
    ).toBeInTheDocument();
  });
});
