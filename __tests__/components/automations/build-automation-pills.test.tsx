import { describe, expect, it, vi } from "vitest";
import { HttpError } from "@openhands/typescript-client";
import {
  extractDraftDispatchErrors,
  getDraftExecutionStatusText,
  formFromServerDraft,
  setupDraftFromServerDraft,
} from "#/components/features/automations/setup/automation-setup-draft-service";
import { type AutomationSetupFormValues } from "#/api/automation-setup-types";
import { AutomationRunStatus, type Automation } from "#/types/automation";
import { I18nKey } from "#/i18n/declaration";
import { type AutomationDraftApiResponse } from "#/manifests/types";
import { render, screen, fireEvent } from "@testing-library/react";
import { buildAutomationMetadataPills } from "#/components/features/automations/build-automation-pills";
import { type SkillCardPill } from "#/components/features/skills/skill-card-pill-row";
import userEvent from "@testing-library/user-event";
import { ErrorState } from "#/components/features/automations/error-state";
import { KebabMenu } from "#/components/features/automations/kebab-menu";
import { SearchInput } from "#/components/features/automations/search-input";

describe("automation-setup-draft-service", () => {
  const detailEnvelope = {
    detail: {
      message: "Draft is not dispatchable",
      errors: [
        {
          field: "prompt",
          code: "required",
          message: "Prompt is required",
        },
      ],
    },
  };

  describe("extractDraftDispatchErrors", () => {
    it("reads FastAPI detail envelopes from axios draft dispatch errors", () => {
      const error = {
        isAxiosError: true,
        response: { data: detailEnvelope },
      };

      expect(extractDraftDispatchErrors(error)).toBe("Prompt is required");
    });

    it("reads FastAPI detail envelopes from SDK HttpError draft dispatch errors", () => {
      const error = new HttpError(422, "Unprocessable Entity", detailEnvelope);

      expect(extractDraftDispatchErrors(error)).toBe("Prompt is required");
    });

    it("falls back to the detail message when no field errors are present", () => {
      const error = new HttpError(422, "Unprocessable Entity", {
        detail: { message: "Draft is not dispatchable", errors: [] },
      });

      expect(extractDraftDispatchErrors(error)).toBe(
        "Draft is not dispatchable",
      );
    });
  });

  const baseSavedDraft: AutomationDraftApiResponse = {
    id: "draft-1",
    endpoint: "/v1/preset/prompt",
    name: "Saved draft",
    draft: {},
    validationErrors: null,
    dispatchable: true,
    sourceAutomationId: null,
    materializedAutomationId: null,
    lastTestRunId: null,
    createdAt: "2026-01-01T00:00:00.000Z",
    updatedAt: "2026-01-01T00:00:00.000Z",
  };

  describe("getDraftExecutionStatusText", () => {
    const run = {
      id: "run-1",
      status: AutomationRunStatus.COMPLETED,
      conversation_id: null,
      bash_command_id: null,
      error_detail: null,
      started_at: "2026-01-01T00:00:00Z",
      completed_at: "2026-01-01T00:01:00Z",
    };
    const translate = (key: I18nKey) => String(key);

    it.each([
      AutomationRunStatus.COMPLETED,
      AutomationRunStatus.FAILED,
      AutomationRunStatus.RUNNING,
    ])(
      "shows the current draft error ahead of a historical %s test",
      (status) => {
        const invalidDraft = {
          ...baseSavedDraft,
          dispatchable: false,
          validationErrors: [
            {
              field: "trigger",
              code: "invalid",
              message: "Invalid cron expression: every day",
            },
          ],
        };

        const text = getDraftExecutionStatusText(
          invalidDraft,
          [{ ...run, status }],
          translate,
        );

        expect(text).toBe("Invalid cron expression: every day");
      },
    );

    it("keeps the successful test status for a currently valid draft", () => {
      expect(
        getDraftExecutionStatusText(baseSavedDraft, [run], translate),
      ).toBe(I18nKey.AUTOMATION_SETUP$LATEST_TEST_PASSED);
    });
  });

  const baseForm: AutomationSetupFormValues = {
    name: "",
    prompt: "",
    repository: "",
    pluginSource: "",
    pluginRef: "",
    pluginList: "",
    customCode: "",
    entrypoint: "",
    setupScriptPath: "",
    setupScript: "",
    triggerKind: "cron",
    frequency: "daily",
    time: "09:00",
    scheduleDateTime: "",
    timezone: "UTC",
    weekday: "1",
    customSchedule: "",
    eventSource: "github",
    eventKey: "",
    eventFilter: "",
    model: "",
    agentProfileId: "",
    showTimeout: false,
    timeoutSeconds: "",
    kind: "prompt",
  };

  describe("setupDraftFromServerDraft", () => {
    it("hydrates all repositories and event keys from saved drafts", () => {
      const saved = {
        ...baseSavedDraft,
        draft: {
          prompt: "Triage issues",
          repos: [{ url: "acme/one" }, { url: "acme/two" }],
          trigger: {
            type: "event",
            source: "github",
            on: ["pull_request", "issues"],
          },
        },
      } satisfies AutomationDraftApiResponse;

      const draft = setupDraftFromServerDraft(saved);

      expect(draft.form?.repository).toBe("acme/one, acme/two");
      expect(draft.form?.eventKey).toBe("pull_request,issues");
    });

    it("updates an existing form with all repositories and event keys", () => {
      const saved = {
        ...baseSavedDraft,
        draft: {
          repos: [{ url: "acme/one" }, { url: "acme/two" }],
          trigger: {
            type: "event",
            source: "github",
            on: ["push", "release.published"],
          },
        },
      } satisfies AutomationDraftApiResponse;

      const form = formFromServerDraft(saved, baseForm);

      expect(form.repository).toBe("acme/one, acme/two");
      expect(form.eventKey).toBe("push,release.published");
    });
  });
});

describe("build-automation-pills", () => {
  function buildAutomation(overrides: Partial<Automation> = {}): Automation {
    return {
      id: "automation-1",
      name: "Triage",
      prompt: "Triage the issue.",
      enabled: true,
      trigger: { type: "event", on: "issue.updated", source: "linear" },
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
      ...overrides,
    };
  }

  function renderPills(pills: SkillCardPill[]) {
    render(
      <div>
        {pills.map((pill) => (
          <span key={pill.id} data-testid={`pill-${pill.id}`}>
            {pill.node}
          </span>
        ))}
      </div>,
    );
  }

  describe("buildAutomationMetadataPills", () => {
    it("puts the event and source on separate pills", () => {
      const pills = buildAutomationMetadataPills(buildAutomation(), "unused");

      expect(pills.map((pill) => pill.id)).toEqual([
        "event-trigger",
        "event-source",
      ]);

      renderPills(pills);

      expect(screen.getByTestId("pill-event-trigger")).toHaveTextContent(
        "issue.updated",
      );
      expect(screen.getByTestId("pill-event-trigger")).not.toHaveTextContent(
        "linear",
      );
      expect(screen.getByTestId("pill-event-source")).toHaveTextContent(
        "Linear",
      );
      expect(
        screen.getByTestId("pill-event-source").firstElementChild,
      ).toHaveClass("py-0.5");
      expect(screen.getByTestId("automation-source-logo")).toBeInTheDocument();
    });

    it("renders a fallback icon when the source is not in the catalog", () => {
      renderPills(
        buildAutomationMetadataPills(
          buildAutomation({
            trigger: {
              type: "event",
              on: "alert.fired",
              source: "custom-pager",
            },
          }),
          "unused",
        ),
      );

      expect(screen.getByTestId("pill-event-source")).toHaveTextContent(
        "Custom-Pager",
      );
      expect(screen.getByTestId("automation-source-logo")).toBeInTheDocument();
    });

    it("shows each repository the automation clones, with its full address on hover", () => {
      const pills = buildAutomationMetadataPills(
        buildAutomation({
          trigger: { type: "cron", schedule: "0 9 * * *" },
          repositories: [
            { url: "acme/widgets", provider: "github" },
            { url: "https://github.com/acme/docs" },
            {
              url: "https://bitbucket.example.com/scm/DEM/demo-app.git",
              provider: "bitbucket_data_center",
            },
          ],
        }),
        "unused",
      );

      renderPills(pills);

      expect(screen.getByTestId("pill-repository")).toHaveTextContent(
        "acme/widgets",
      );
      expect(screen.getByTestId("pill-repository-1")).toHaveTextContent(
        "acme/docs",
      );
      expect(
        screen.getByTitle("https://github.com/acme/docs"),
      ).toBeInTheDocument();
      expect(screen.getByTestId("pill-repository-2").textContent).toBe(
        "DEM/demo-app",
      );
    });

    it("omits the source pill when the event has no source", () => {
      const pills = buildAutomationMetadataPills(
        buildAutomation({
          trigger: { type: "event", on: "pull_request.opened" },
        }),
        "unused",
      );

      expect(pills.map((pill) => pill.id)).toEqual(["event-trigger"]);
    });
  });
});

describe("ErrorState", () => {
  it("renders error icon and retry button", () => {
    const { container } = render(<ErrorState onRetry={vi.fn()} />);

    // Check for the error icon (ExclamationCircleIcon)
    expect(container.querySelector("svg")).toBeInTheDocument();
    // Check for the retry button
    expect(screen.getByRole("button")).toBeInTheDocument();
  });

  it("calls onRetry when retry button is clicked", async () => {
    const onRetry = vi.fn();
    const user = userEvent.setup();
    render(<ErrorState onRetry={onRetry} />);

    await user.click(screen.getByRole("button"));

    expect(onRetry).toHaveBeenCalledOnce();
  });
});

describe("kebab-menu", () => {
  const ITEMS = ["Run now", "View", "Export", "Edit", "Turn on", "Delete"].map(
    (label) => ({ label, icon: <span />, onClick: vi.fn() }),
  );

  /** Place the trigger at a fixed viewport position before the menu opens. */
  function openMenuWithTriggerAt(top: number, bottom: number) {
    const trigger = screen.getByRole("button", {
      name: "AUTOMATIONS$ACTIONS_MENU",
    });
    vi.spyOn(trigger, "getBoundingClientRect").mockReturnValue({
      top,
      bottom,
      left: 900,
      right: 1000,
      width: 100,
      height: bottom - top,
      x: 900,
      y: top,
      toJSON: () => ({}),
    } as DOMRect);
    fireEvent.click(trigger);
    return document.querySelector<HTMLElement>('div[style*="position: fixed"]');
  }

  describe("KebabMenu", () => {
    it("opens the menu and invokes an item's onClick when selected", async () => {
      const onClick = vi.fn();
      render(
        <KebabMenu items={[{ label: "Delete", icon: <span />, onClick }]} />,
      );

      // Open the menu, then select the item (rendered via KebabMenuItemContent).
      fireEvent.click(
        screen.getByRole("button", { name: "AUTOMATIONS$ACTIONS_MENU" }),
      );
      fireEvent.click(await screen.findByText("Delete"));

      expect(onClick).toHaveBeenCalledTimes(1);
    });

    it("anchors below the trigger when the menu fits in the viewport", () => {
      render(<KebabMenu items={ITEMS} />);

      const portal = openMenuWithTriggerAt(68, 100);

      expect(portal?.style.top).toBe("104px");
      expect(portal?.style.bottom).toBe("");
    });

    it("flips above the trigger when the menu would clip at the viewport bottom", () => {
      render(<KebabMenu items={ITEMS} />);

      // window.innerHeight is 768 in jsdom; 700 + 4 + (6 * 36) overflows it.
      const portal = openMenuWithTriggerAt(668, 700);

      expect(portal?.style.bottom).toBe("104px");
      expect(portal?.style.top).toBe("");
    });
  });
});

describe("SearchInput", () => {
  it("calls onChange when user types", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<SearchInput value="" onChange={onChange} />);

    const input = screen.getByRole("textbox");
    await user.type(input, "test");

    expect(onChange).toHaveBeenCalled();
    expect(onChange).toHaveBeenLastCalledWith("t");
  });

  it("displays the current value", () => {
    render(<SearchInput value="security" onChange={vi.fn()} />);

    expect(screen.getByRole("textbox")).toHaveValue("security");
  });
});
