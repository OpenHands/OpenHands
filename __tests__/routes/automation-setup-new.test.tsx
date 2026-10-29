import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AutomationSetupNew from "#/routes/automation-setup-new";
import AutomationService from "#/api/automation-service/automation-service.api";
import {
  PENDING_AUTOMATION_SETUP_ID,
  clearAutomationFormSession,
} from "#/api/automation-form-session";
import type { Automation } from "#/types/automation";
import type { AutomationDraftApiResponse } from "#/manifests/types";

const { mockStartConversationFromPrompt } = vi.hoisted(() => ({
  mockStartConversationFromPrompt: vi.fn(),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    getAutomation: vi.fn(),
    getServerDraft: vi.fn(),
  },
}));

vi.mock("#/hooks/use-start-automation-setup", () => ({
  useStartAutomationSetup: () => ({
    startConversationFromPrompt: mockStartConversationFromPrompt,
    isPending: false,
  }),
}));

vi.mock("#/components/features/chat/interactive-chat-box", () => ({
  InteractiveChatBox: () => <div data-testid="mock-chat-box" />,
}));

vi.mock(
  "#/components/features/automations/setup/automation-setup-panel",
  () => ({
    AutomationSetupPanel: ({
      draft,
      onAskAgent,
    }: {
      draft: unknown;
      onAskAgent?: (prompt: string) => void;
    }) => {
      const askAgentLabel = "ask agent";
      return (
        <>
          <pre data-testid="setup-panel-draft">{JSON.stringify(draft)}</pre>
          <button
            type="button"
            data-testid="mock-ask-agent"
            onClick={() => onAskAgent?.("ASK_AGENT_PROMPT")}
          >
            {askAgentLabel}
          </button>
        </>
      );
    },
  }),
);

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
}));

const automation: Automation = {
  id: "auto-1",
  name: "Daily digest",
  prompt: "Review pull requests",
  enabled: true,
  created_at: "2026-01-01T00:00:00.000Z",
  updated_at: "2026-01-01T00:00:00.000Z",
  repository: "OpenHands/OpenHands",
  trigger: { type: "cron", schedule: "0 9 * * *", timezone: "UTC" },
};

const serverDraft: AutomationDraftApiResponse = {
  id: "draft-1",
  endpoint: "/v1/preset/prompt",
  name: "Saved draft",
  draft: {
    prompt: "Triage issues",
    repos: [{ url: "acme/one" }, { url: "acme/two" }],
    trigger: { type: "event", source: "github", on: ["issues", "push"] },
  },
  validationErrors: null,
  dispatchable: true,
  sourceAutomationId: null,
  materializedAutomationId: null,
  lastTestRunId: null,
  createdAt: "2026-01-01T00:00:00.000Z",
  updatedAt: "2026-01-01T00:00:00.000Z",
};

function renderRoute(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <AutomationSetupNew />
    </MemoryRouter>,
  );
}

describe("AutomationSetupNew", () => {
  beforeEach(() => {
    clearAutomationFormSession(PENDING_AUTOMATION_SETUP_ID);
    vi.clearAllMocks();
  });

  it("reloads an existing automation editor from the automationId query", async () => {
    vi.mocked(AutomationService.getAutomation).mockResolvedValue(automation);

    renderRoute("/automations/setup?automationId=auto-1");

    await waitFor(() =>
      expect(AutomationService.getAutomation).toHaveBeenCalledWith("auto-1"),
    );
    const draft = JSON.parse(
      screen.getByTestId("setup-panel-draft").textContent ?? "{}",
    );
    expect(draft.editingAutomationId).toBe("auto-1");
    expect(draft.form.name).toBe("Daily digest");
  });

  it("reloads a saved draft from the draftId query", async () => {
    vi.mocked(AutomationService.getServerDraft).mockResolvedValue(serverDraft);

    renderRoute("/automations/setup?draftId=draft-1");

    await waitFor(() =>
      expect(AutomationService.getServerDraft).toHaveBeenCalledWith("draft-1"),
    );
    const draft = JSON.parse(
      screen.getByTestId("setup-panel-draft").textContent ?? "{}",
    );
    expect(draft.serverDraftId).toBe("draft-1");
    expect(draft.form.repository).toBe("acme/one, acme/two");
    expect(draft.form.eventKey).toBe("issues,push");
  });

  it("hands the Ask Agent prompt to the setup conversation starter", async () => {
    renderRoute("/automations/setup");

    fireEvent.click(await screen.findByTestId("mock-ask-agent"));

    expect(mockStartConversationFromPrompt).toHaveBeenCalledWith(
      "ASK_AGENT_PROMPT",
    );
  });
});
