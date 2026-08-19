import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AutomationService from "#/api/automation-service/automation-service.api";
import { SetupDialog } from "#/components/features/manifest/manifest-setup-dialog";
import type { SetupPrerequisitesResult } from "#/hooks/query/use-manifest-prerequisites";
import type { SetupEntry } from "#/manifests/types";
import {
  createSetup,
  createSetupEntry,
} from "../../manifests/manifest-test-data";

/**
 * The dialog is the part of setup that is not pure: it owns the step order, the
 * local check that has to pass before the service is asked anything, what the
 * action is handed, and where a finished setup lands. Each stage it drives has
 * its own test, so those are stubbed here and only the wiring is exercised.
 */
const mocks = vi.hoisted(() => ({
  navigate: vi.fn(),
  runAction: vi.fn(),
  prerequisites: vi.fn(),
  capabilities: vi.fn(),
  missingCreateEndpoints: vi.fn<(entry: SetupEntry) => string[]>(() => []),
  tracking: {
    trackAutomationSetupOpened: vi.fn(),
    trackAutomationSetupValidated: vi.fn(),
    trackAutomationSetupCreated: vi.fn(),
    trackAutomationSetupFailed: vi.fn(),
  },
}));

vi.mock("react-router", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-router")>()),
  useNavigate: () => mocks.navigate,
}));

// A local backend, where the repository field is a plain input rather than the
// picker only a cloud backend can populate.
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: "local-1", kind: "local" },
    orgId: null,
  }),
}));

vi.mock("#/hooks/query/use-manifest-capabilities", () => ({
  useSetupCapabilities: () => mocks.capabilities(),
}));

vi.mock("#/hooks/query/use-manifest-prerequisites", () => ({
  useSetupPrerequisites: () => mocks.prerequisites(),
}));

// Which endpoints an entry cannot be created without is read off the published
// interface manifest, so a real one that declares them leaves the refusal path
// unreachable. Stubbed so the case states the manifest it is about, rather than
// depending on the packaged manifest continuing not to publish them.
vi.mock("#/manifests/automation-setup", async (importOriginal) => ({
  ...(await importOriginal<typeof import("#/manifests/automation-setup")>()),
  missingCreateEndpoints: mocks.missingCreateEndpoints,
}));

vi.mock("#/manifests/manifest-actions", () => ({
  useSetupAction: () => mocks.runAction,
}));

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => mocks.tracking,
}));

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: { validateDraft: vi.fn() },
}));

const NOTHING_TO_CONNECT: SetupPrerequisitesResult = {
  blockingIntegrations: [],
  warningIntegrations: [],
  isBlocked: false,
  isLoading: false,
};

const ENTRY: SetupEntry = createSetupEntry();

function renderDialog(entry: SetupEntry = ENTRY) {
  const user = userEvent.setup();
  render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <SetupDialog entry={entry} onClose={vi.fn()} />
    </QueryClientProvider>,
  );
  return { user };
}

/** Answer the two required fields the entry ships without a default. */
async function fillForm(user: ReturnType<typeof userEvent.setup>) {
  await user.type(
    screen.getByTestId("setup-field-repository"),
    "OpenHands/agent-server-gui",
  );
  await user.type(screen.getByTestId("setup-field-widgetName"), "Widgets");
}

beforeEach(() => {
  vi.clearAllMocks();
  // clearAllMocks resets calls, not implementations, so the one case that
  // stubs a manifest without the bundle endpoints would leak into the rest.
  mocks.missingCreateEndpoints.mockReturnValue([]);
  mocks.prerequisites.mockReturnValue(NOTHING_TO_CONNECT);
  mocks.capabilities.mockReturnValue({
    capabilities: null,
    supported: "unknown",
    unmet: [],
    isLoading: false,
  });
  vi.mocked(AutomationService.validateDraft).mockResolvedValue({
    valid: true,
    errors: [],
  });
});

/** The same entry once it asks for several repositories. */
const MULTI_REPO_ENTRY: SetupEntry = (() => {
  const { form } = createSetup();
  return createSetupEntry({
    setup: createSetup({
      form: {
        ...form,
        args: {
          ...form.args,
          repository: { ...form.args.repository, multiple: true },
        },
      },
    }),
  });
})();

/** An entry that ships a script bundle rather than a prompt. */
const BUNDLE_ENTRY: SetupEntry = createSetupEntry({
  setup: createSetup({
    prompt: undefined,
    bundle: {
      version: "1.0.0",
      entrypoint: "python3 main.py",
      files: { "main.py": "skills/widget-monitor/scripts/main.py" },
      config: { repos: ["{{form.repository}}"] },
    },
  }),
});

/** A deployment that answered discovery and came up short. */
const UNSUPPORTED = {
  capabilities: null,
  supported: false as const,
  unmet: ["webhookDelivery"],
  isLoading: false,
};

describe("SetupDialog", () => {
  it("asks about an unconnected integration before it asks anything else", async () => {
    // Arrange — an advisory integration, which is shown but does not block.
    mocks.prerequisites.mockReturnValue({
      ...NOTHING_TO_CONNECT,
      warningIntegrations: [
        {
          id: "github",
          requirement: { message: "Used to read widgets.", required: false },
          entry: null,
        },
      ],
    });
    const { user } = renderDialog();

    // Act
    expect(screen.getByTestId("setup-prerequisites")).toBeInTheDocument();
    await user.click(screen.getByTestId("setup-continue-button"));

    // Assert
    expect(screen.getByTestId("setup-field-widgetName")).toBeInTheDocument();
    expect(screen.queryByTestId("setup-prerequisites")).toBeNull();
  });

  it("holds an unanswered required field back from the service", async () => {
    // Arrange — nothing typed, so two required fields are still empty.
    const { user } = renderDialog();

    // Act
    await user.click(screen.getByTestId("setup-continue-button"));

    // Assert — the local check reports the field's own code rather than
    // collapsing every failure into one message, and nothing was sent.
    expect(
      screen.getByTestId("setup-field-widgetName-error"),
    ).toHaveTextContent("SETUP$VALIDATION_REQUIRED");
    expect(AutomationService.validateDraft).not.toHaveBeenCalled();
    expect(screen.queryByTestId("setup-review")).toBeNull();
  });

  it("creates from the derived payload and opens what was created", async () => {
    // Arrange
    mocks.runAction.mockResolvedValue({ response: { id: "automation-1" } });
    const { user } = renderDialog();
    await fillForm(user);

    // Act
    await user.click(screen.getByTestId("setup-continue-button"));
    await waitFor(() =>
      expect(screen.getByTestId("setup-review")).toBeInTheDocument(),
    );
    await user.click(screen.getByTestId("setup-continue-button"));

    // Assert — the action is handed the derived request body, not the raw
    // answers, and the new automation is where setup lands.
    await waitFor(() =>
      expect(mocks.navigate).toHaveBeenCalledWith("/automations/automation-1", {
        replace: true,
      }),
    );
    expect(mocks.runAction.mock.calls[0][2]).toEqual({
      name: "Widget monitor - OpenHands/agent-server-gui",
      prompt: "Report on Widgets in OpenHands/agent-server-gui.",
      repos: [{ url: "OpenHands/agent-server-gui", provider: "github" }],
      trigger: { type: "cron", schedule: "*/15 * * * *" },
    });
  });

  it("offers the conversation fallback when the deployment cannot run a direct entry", async () => {
    // Arrange — capabilities answered and came up short, and the entry ships
    // a fallback-conversation seed.
    mocks.capabilities.mockReturnValue(UNSUPPORTED);
    mocks.runAction.mockResolvedValue({
      response: { conversation_id: "conv-1" },
    });
    const entry = createSetupEntry({
      setup: createSetup({
        message: "Set this up in a conversation instead.",
      }),
    });
    const { user } = renderDialog(entry);

    // Act
    await user.click(screen.getByTestId("setup-fallback-conversation"));

    // Assert — the action runs with no payload, the assisted outcome, and
    // setup lands in the conversation that will finish it.
    await waitFor(() =>
      expect(mocks.navigate).toHaveBeenCalledWith("/conversations/conv-1", {
        replace: true,
      }),
    );
    expect(mocks.runAction).toHaveBeenCalledWith(
      entry,
      expect.anything(),
      null,
    );
  });

  it("keeps the unsupported screen close-only when there is nothing to fall back to", () => {
    // Arrange — no skill command resolves for this entry and it declares no
    // fallback message, so a conversation would open empty-handed.
    mocks.capabilities.mockReturnValue(UNSUPPORTED);
    renderDialog();

    // Assert
    expect(screen.queryByTestId("setup-fallback-conversation")).toBeNull();
  });

  it("carries a repository typed but not added through to the review step", async () => {
    // Arrange — the list is built by adding entries, and the input still shows
    // what was typed when the user reaches for Continue.
    const { user } = renderDialog(MULTI_REPO_ENTRY);
    await user.type(screen.getByTestId("setup-field-widgetName"), "Widgets");
    await user.type(
      screen.getByTestId("setup-field-repository"),
      "OpenHands/automation",
    );

    // Act — Continue, without pressing Add or Enter first.
    await user.click(screen.getByTestId("setup-continue-button"));

    // Assert — the answer the user could still see is the one being confirmed.
    await waitFor(() =>
      expect(screen.getByTestId("setup-review")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("setup-review")).toHaveTextContent(
      "OpenHands/automation",
    );
  });

  it("refuses an entry the published interface declares no way to create", async () => {
    // Arrange — a bundle entry against an interface manifest published before
    // bundles: neither endpoint it needs exists, and no answer supplies them.
    mocks.missingCreateEndpoints.mockReturnValue(["createBundle", "uploads"]);
    renderDialog(BUNDLE_ENTRY);

    // Assert — said before the form, rather than as a Continue button that
    // silently does nothing once the form is filled in.
    expect(screen.getByTestId("setup-unmet-requirements")).toHaveTextContent(
      "createBundle, uploads",
    );
    expect(screen.queryByTestId("setup-field-widgetName")).toBeNull();
  });

  it("returns a rejected create to the field the service blamed", async () => {
    // Arrange — a validation failure addressed by payload path, which only the
    // derived error map can turn back into a field.
    mocks.runAction.mockRejectedValue(
      Object.assign(new Error("Unprocessable Entity"), {
        name: "HttpError",
        status: 422,
        response: {
          detail: [
            {
              loc: ["body", "trigger", "cron", "schedule"],
              msg: "Interval is too short",
            },
          ],
        },
      }),
    );
    const { user } = renderDialog();
    await fillForm(user);

    // Act
    await user.click(screen.getByTestId("setup-continue-button"));
    await waitFor(() =>
      expect(screen.getByTestId("setup-review")).toBeInTheDocument(),
    );
    await user.click(screen.getByTestId("setup-continue-button"));

    // Assert — back on the form, with the message against `schedule` rather
    // than against the form as a whole.
    await waitFor(() =>
      expect(
        screen.getByTestId("setup-field-schedule-error"),
      ).toHaveTextContent("Interval is too short"),
    );
    expect(screen.queryByTestId("setup-review")).toBeNull();
    expect(mocks.navigate).not.toHaveBeenCalled();
  });
});
