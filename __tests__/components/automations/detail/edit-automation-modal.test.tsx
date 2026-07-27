import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { EditAutomationModal } from "#/components/features/automations/detail/edit-automation-modal";
import AutomationService from "#/api/automation-service/automation-service.api";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import {
  displaySuccessToast,
  displayErrorToast,
} from "#/utils/custom-toast-handlers";
import type { Automation } from "#/types/automation";
import type { Backend } from "#/api/backend-registry/types";

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    updateAutomation: vi.fn(),
  },
}));

vi.mock("#/api/profiles-service/profiles-service.api", () => ({
  default: {
    listProfiles: vi.fn(),
  },
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: vi.fn(),
  displayErrorToast: vi.fn(),
}));

const localBackend: Backend = {
  id: "local-1",
  name: "Local",
  host: "http://localhost:8000",
  apiKey: "session-key",
  kind: "local",
};

const dailyAutomation: Automation = {
  id: "auto-1",
  name: "Daily digest",
  prompt: "Summarize yesterday's PRs",
  trigger: { type: "cron", schedule: "0 9 * * *" },
  enabled: true,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  timezone: "America/Los_Angeles",
};

const customAutomation: Automation = {
  ...dailyAutomation,
  id: "auto-2",
  name: "Twice daily",
  trigger: { type: "cron", schedule: "0 9,17 * * *" },
};

// A schedule automation pinned to a concrete LLM profile, used to exercise
// the profile picker (the base fixtures intentionally leave `model` unset).
const modeledAutomation: Automation = {
  ...dailyAutomation,
  id: "auto-3",
  model: "fast",
};

// An automation carrying an explicit run timeout (in seconds). The base
// fixtures intentionally leave `timeout` unset.
const timeoutAutomation: Automation = {
  ...dailyAutomation,
  id: "auto-4",
  timeout: 600,
};

const profilesResponse = {
  profiles: [
    {
      name: "fast",
      model: "anthropic/claude-haiku-4-5",
      base_url: null,
      api_key_set: true,
    },
    {
      name: "careful",
      model: "anthropic/claude-opus-4-8",
      base_url: null,
      api_key_set: true,
    },
  ],
  active_profile: "fast",
};

function renderModal(automation: Automation) {
  const onClose = vi.fn();
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const utils = render(
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>
        <EditAutomationModal
          automation={automation}
          isOpen
          onClose={onClose}
        />
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
  return { ...utils, onClose };
}

beforeEach(() => {
  vi.clearAllMocks();
  __resetActiveStoreForTests();
  setRegisteredBackends([localBackend]);
  setActiveSelection({ backendId: localBackend.id });
  // Default to no profiles; tests that exercise the picker override this.
  vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
    profiles: [],
    active_profile: null,
  });
});

describe("EditAutomationModal", () => {
  it("pre-fills current values and PATCHes only the fields that changed", async () => {
    // Arrange — daily automation at 09:00 with a known prompt. The
    // backend will echo back the merged result.
    vi.mocked(AutomationService.updateAutomation).mockResolvedValue({
      ...dailyAutomation,
      name: "Morning digest",
      prompt: "Summarize today's open PRs",
      trigger: { type: "cron", schedule: "30 10 * * *" },
    });
    const user = userEvent.setup();
    const { onClose } = renderModal(dailyAutomation);

    // Sanity-check pre-fill: the inputs reflect the existing automation
    // before the user edits anything.
    const nameInput = screen.getByTestId(
      "edit-automation-name",
    ) as HTMLInputElement;
    const timeInput = screen.getByTestId(
      "edit-automation-time",
    ) as HTMLInputElement;
    expect(nameInput.value).toBe("Daily digest");
    expect(timeInput.value).toBe("09:00");

    // Act — change name, prompt, and time; leave frequency at Daily.
    await user.clear(nameInput);
    await user.type(nameInput, "Morning digest");
    const promptInput = screen.getByTestId("edit-automation-prompt");
    await user.clear(promptInput);
    await user.type(promptInput, "Summarize today's open PRs");
    await user.clear(timeInput);
    await user.type(timeInput, "10:30");
    await user.click(screen.getByTestId("edit-automation-save"));

    // Assert — PATCH body contains exactly the diff (no untouched
    // fields like enabled/repository), and the success path closes
    // the modal + toasts the user.
    await waitFor(() => {
      expect(AutomationService.updateAutomation).toHaveBeenCalledTimes(1);
    });
    expect(AutomationService.updateAutomation).toHaveBeenCalledWith("auto-1", {
      name: "Morning digest",
      prompt: "Summarize today's open PRs",
      trigger: { type: "cron", schedule: "30 10 * * *" },
    });
    await waitFor(() => {
      expect(displaySuccessToast).toHaveBeenCalledTimes(1);
    });
    expect(onClose).toHaveBeenCalled();
  });

  it("blocks submit and shows a validation error when the name is empty", async () => {
    // Arrange
    const user = userEvent.setup();
    renderModal(dailyAutomation);

    // Act — clear the name and try to save.
    const nameInput = screen.getByTestId("edit-automation-name");
    await user.clear(nameInput);
    await user.click(screen.getByTestId("edit-automation-save"));

    // Assert — no PATCH fired, inline error appears.
    expect(AutomationService.updateAutomation).not.toHaveBeenCalled();
    expect(
      screen.getByTestId("edit-automation-name-error"),
    ).toBeInTheDocument();
  });

  it("renders custom-schedule hint and skips schedule mutation for non-preset cron", async () => {
    // Arrange — schedule "0 9,17 * * *" is not a Daily/Weekdays/Weekly
    // preset; the modal must treat it as read-only for frequency but
    // still allow editing the prompt/name.
    vi.mocked(AutomationService.updateAutomation).mockResolvedValue(
      customAutomation,
    );
    const user = userEvent.setup();
    renderModal(customAutomation);

    // The hint surfaces so the user understands why frequency is
    // disabled; this is the user-visible signal that we're in
    // custom mode.
    expect(screen.getByTestId("custom-schedule-hint")).toBeInTheDocument();

    // Act — change only the name and save.
    const nameInput = screen.getByTestId("edit-automation-name");
    await user.clear(nameInput);
    await user.type(nameInput, "Renamed");
    await user.click(screen.getByTestId("edit-automation-save"));

    // Assert — the PATCH body does NOT include a trigger override, so
    // the user's hand-tuned cron is preserved.
    await waitFor(() => {
      expect(AutomationService.updateAutomation).toHaveBeenCalledTimes(1);
    });
    const [, body] = vi.mocked(AutomationService.updateAutomation).mock
      .calls[0];
    expect(body).not.toHaveProperty("trigger");
    expect(body).toMatchObject({ name: "Renamed" });
  });

  it("surfaces an error toast and keeps the modal open when the update fails", async () => {
    // Arrange — backend rejects the PATCH.
    vi.mocked(AutomationService.updateAutomation).mockRejectedValue(
      new Error("backend down"),
    );
    const user = userEvent.setup();
    const { onClose } = renderModal(dailyAutomation);

    // Act — change the time to force a non-empty diff, then save.
    const timeInput = screen.getByTestId("edit-automation-time");
    await user.clear(timeInput);
    await user.type(timeInput, "10:30");
    await user.click(screen.getByTestId("edit-automation-save"));

    // Assert — error toast fired, modal stays open.
    await waitFor(() => {
      expect(displayErrorToast).toHaveBeenCalledTimes(1);
    });
    expect(onClose).not.toHaveBeenCalled();
  });

  it("persists the newly selected LLM profile in the update payload", async () => {
    // Arrange — automation currently runs on the "fast" profile, with a
    // second "careful" profile available to switch to.
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue(profilesResponse);
    vi.mocked(AutomationService.updateAutomation).mockResolvedValue({
      ...modeledAutomation,
      model: "careful",
    });
    const user = userEvent.setup();
    const { onClose } = renderModal(modeledAutomation);

    // The picker pre-fills with the automation's current profile once the
    // available profiles have loaded.
    await waitFor(() =>
      expect(screen.getByLabelText("AUTOMATIONS$DETAIL$MODEL")).toHaveValue(
        "fast",
      ),
    );

    // Act — switch to "careful" and save.
    await user.click(screen.getByLabelText("AUTOMATIONS$DETAIL$MODEL"));
    await user.click(await screen.findByText("careful"));
    await user.click(screen.getByTestId("edit-automation-save"));

    // Assert — only the profile changed, so the PATCH carries just `model`.
    await waitFor(() => {
      expect(AutomationService.updateAutomation).toHaveBeenCalledTimes(1);
    });
    expect(AutomationService.updateAutomation).toHaveBeenCalledWith("auto-3", {
      model: "careful",
    });
    expect(onClose).toHaveBeenCalled();
  });

  it("resets the LLM profile to active when 'Active profile' is selected", async () => {
    // Arrange — automation pinned to "fast"; the backend re-resolves a null
    // model back to whatever profile is active.
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue(profilesResponse);
    vi.mocked(AutomationService.updateAutomation).mockResolvedValue({
      ...modeledAutomation,
      model: "fast",
    });
    const user = userEvent.setup();
    renderModal(modeledAutomation);

    // The picker pre-fills with the pinned profile once profiles have loaded.
    await waitFor(() =>
      expect(screen.getByLabelText("AUTOMATIONS$DETAIL$MODEL")).toHaveValue(
        "fast",
      ),
    );

    // Act — clear the pin via the "Active profile" option, then save.
    await user.click(screen.getByLabelText("AUTOMATIONS$DETAIL$MODEL"));
    await user.click(await screen.findByText("COMMON$ACTIVE_PROFILE"));
    await user.click(screen.getByTestId("edit-automation-save"));

    // Assert — the PATCH carries model: null so the backend resets to active.
    await waitFor(() => {
      expect(AutomationService.updateAutomation).toHaveBeenCalledTimes(1);
    });
    expect(AutomationService.updateAutomation).toHaveBeenCalledWith("auto-3", {
      model: null,
    });
  });

  it("omits the LLM profile from the payload when it is left unchanged", async () => {
    // Arrange — profiles available; the user will only rename the automation.
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue(profilesResponse);
    vi.mocked(AutomationService.updateAutomation).mockResolvedValue(
      modeledAutomation,
    );
    const user = userEvent.setup();
    renderModal(modeledAutomation);

    // Ensure we're on the profiles-available path before editing.
    await screen.findByLabelText("AUTOMATIONS$DETAIL$MODEL");

    // Act — change only the name; leave the profile on "fast".
    const nameInput = screen.getByTestId("edit-automation-name");
    await user.clear(nameInput);
    await user.type(nameInput, "Renamed digest");
    await user.click(screen.getByTestId("edit-automation-save"));

    // Assert — the PATCH renames but does not resend the unchanged profile.
    await waitFor(() => {
      expect(AutomationService.updateAutomation).toHaveBeenCalledTimes(1);
    });
    const [, body] = vi.mocked(AutomationService.updateAutomation).mock
      .calls[0];
    expect(body).toMatchObject({ name: "Renamed digest" });
    expect(body).not.toHaveProperty("model");
  });

  it("hides the LLM profile picker when no profiles are available", async () => {
    // Arrange — beforeEach already mocks an empty profile list.
    renderModal(dailyAutomation);

    // Assert — once the (empty) profile list resolves, no picker is offered.
    await waitFor(() => {
      expect(
        screen.queryByLabelText("AUTOMATIONS$DETAIL$MODEL"),
      ).not.toBeInTheDocument();
    });
  });

  it("pre-fills the timeout and sends the new value when it changes", async () => {
    // Arrange — automation currently times out after 600s.
    vi.mocked(AutomationService.updateAutomation).mockResolvedValue({
      ...timeoutAutomation,
      timeout: 1800,
    });
    const user = userEvent.setup();
    renderModal(timeoutAutomation);

    // Sanity-check pre-fill before editing.
    const timeoutInput = screen.getByTestId(
      "edit-automation-timeout",
    ) as HTMLInputElement;
    expect(timeoutInput.value).toBe("600");

    // Act — raise the timeout to the 1800s maximum and save.
    await user.clear(timeoutInput);
    await user.type(timeoutInput, "1800");
    await user.click(screen.getByTestId("edit-automation-save"));

    // Assert — the PATCH carries just the new timeout.
    await waitFor(() => {
      expect(AutomationService.updateAutomation).toHaveBeenCalledTimes(1);
    });
    expect(AutomationService.updateAutomation).toHaveBeenCalledWith("auto-4", {
      timeout: 1800,
    });
  });

  it("sends timeout: null when the timeout field is cleared", async () => {
    // Arrange — automation has an explicit 600s timeout.
    vi.mocked(AutomationService.updateAutomation).mockResolvedValue({
      ...timeoutAutomation,
      timeout: null,
    });
    const user = userEvent.setup();
    renderModal(timeoutAutomation);

    // Act — clear the field to fall back to the server default, then save.
    await user.clear(screen.getByTestId("edit-automation-timeout"));
    await user.click(screen.getByTestId("edit-automation-save"));

    // Assert — the PATCH resets the stored timeout to null.
    await waitFor(() => {
      expect(AutomationService.updateAutomation).toHaveBeenCalledTimes(1);
    });
    expect(AutomationService.updateAutomation).toHaveBeenCalledWith("auto-4", {
      timeout: null,
    });
  });

  it("blocks submit and shows a validation error for an out-of-range timeout", async () => {
    // Arrange
    const user = userEvent.setup();
    renderModal(timeoutAutomation);

    // Act — enter a timeout beyond the 1800s cap and try to save.
    const timeoutInput = screen.getByTestId("edit-automation-timeout");
    await user.clear(timeoutInput);
    await user.type(timeoutInput, "2000");
    await user.click(screen.getByTestId("edit-automation-save"));

    // Assert — no PATCH fired, inline error appears.
    expect(AutomationService.updateAutomation).not.toHaveBeenCalled();
    expect(
      screen.getByTestId("edit-automation-timeout-error"),
    ).toBeInTheDocument();
  });

  it("omits the timeout from the payload when it is left unchanged", async () => {
    // Arrange
    vi.mocked(AutomationService.updateAutomation).mockResolvedValue(
      timeoutAutomation,
    );
    const user = userEvent.setup();
    renderModal(timeoutAutomation);

    // Act — rename the automation but leave the timeout at 600.
    const nameInput = screen.getByTestId("edit-automation-name");
    await user.clear(nameInput);
    await user.type(nameInput, "Renamed digest");
    await user.click(screen.getByTestId("edit-automation-save"));

    // Assert — the PATCH renames but does not resend the unchanged timeout.
    await waitFor(() => {
      expect(AutomationService.updateAutomation).toHaveBeenCalledTimes(1);
    });
    const [, body] = vi.mocked(AutomationService.updateAutomation).mock
      .calls[0];
    expect(body).toMatchObject({ name: "Renamed digest" });
    expect(body).not.toHaveProperty("timeout");
  });
});
