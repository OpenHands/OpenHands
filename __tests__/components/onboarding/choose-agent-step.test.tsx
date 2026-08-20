import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  AgentOptionIcon,
  ChooseAgentStep,
  type OnboardingAgentId,
} from "#/components/features/onboarding/steps/choose-agent-step";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  ACP_PROVIDERS,
  getAcpPreferredDefaultModel,
} from "#/constants/acp-providers";
import { I18nKey } from "#/i18n/declaration";

function renderStep(initial: OnboardingAgentId = "openhands") {
  const onSelect = vi.fn();
  const onNext = vi.fn();
  render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <ChooseAgentStep
        selectedAgentId={initial}
        onSelect={onSelect}
        onNext={onNext}
      />
    </QueryClientProvider>,
  );
  return { onSelect, onNext };
}

describe("ChooseAgentStep", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);
  });

  it("renders all four agent options with OpenHands marked selected by default", () => {
    renderStep();

    const openhands = screen.getByTestId("onboarding-agent-option-openhands");
    const claude = screen.getByTestId("onboarding-agent-option-claude-code");
    const codex = screen.getByTestId("onboarding-agent-option-codex");
    const gemini = screen.getByTestId("onboarding-agent-option-gemini-cli");

    expect(openhands).toHaveAttribute("aria-checked", "true");
    // All four options are clickable — ACP is no longer "coming soon".
    expect(openhands).not.toBeDisabled();
    expect(claude).not.toBeDisabled();
    expect(codex).not.toBeDisabled();
    expect(gemini).not.toBeDisabled();

    // Neither the legacy "coming soon" banner nor the per-option badges
    // should render now that all four agent kinds work end-to-end.
    expect(
      screen.queryByTestId("onboarding-agent-coming-soon"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("onboarding-agent-badge-claude-code"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("onboarding-agent-badge-codex"),
    ).not.toBeInTheDocument();
  });

  it("renders the registry-selected Gemini icon on the Gemini CLI tile", () => {
    renderStep();

    const gemini = screen.getByTestId("onboarding-agent-option-gemini-cli");

    expect(
      within(gemini).getByTestId("onboarding-agent-icon-gemini"),
    ).toBeInTheDocument();
    expect(
      within(gemini).queryByTestId("onboarding-agent-icon-codex"),
    ).not.toBeInTheDocument();
  });

  it("falls back to the generic CLI icon for a registry provider without an icon", () => {
    ACP_PROVIDERS.push({
      key: "future-cli",
      display_name: "Future CLI",
      default_command: ["future-cli", "--acp"],
      description_key: I18nKey.ONBOARDING$AGENT_CODEX_DESCRIPTION,
    });

    try {
      renderStep();

      const future = screen.getByTestId("onboarding-agent-option-future-cli");

      expect(
        within(future).getByTestId("onboarding-agent-icon-cli-generic"),
      ).toBeInTheDocument();
      expect(
        within(future).queryByTestId("onboarding-agent-icon-codex"),
      ).not.toBeInTheDocument();
    } finally {
      ACP_PROVIDERS.pop();
    }
  });

  it("falls back to the generic CLI icon for an unknown provider key", () => {
    render(<AgentOptionIcon id="unknown-cli" muted={false} />);

    expect(
      screen.getByTestId("onboarding-agent-icon-cli-generic"),
    ).toBeInTheDocument();
  });

  it("propagates click selections through onSelect for every option", async () => {
    const { onSelect } = renderStep();
    const user = userEvent.setup();

    await user.click(screen.getByTestId("onboarding-agent-option-claude-code"));
    expect(onSelect).toHaveBeenLastCalledWith("claude-code");

    await user.click(screen.getByTestId("onboarding-agent-option-codex"));
    expect(onSelect).toHaveBeenLastCalledWith("codex");

    await user.click(screen.getByTestId("onboarding-agent-option-gemini-cli"));
    expect(onSelect).toHaveBeenLastCalledWith("gemini-cli");

    await user.click(screen.getByTestId("onboarding-agent-option-openhands"));
    expect(onSelect).toHaveBeenLastCalledWith("openhands");
  });

  it("persists agent_kind:'openhands' and advances on Next when OpenHands is selected", async () => {
    const save = vi.spyOn(SettingsService, "saveSettings");
    const { onNext } = renderStep("openhands");
    const user = userEvent.setup();

    await user.click(screen.getByTestId("onboarding-agent-next"));

    await waitFor(() => {
      expect(save).toHaveBeenCalledTimes(1);
      expect(onNext).toHaveBeenCalledTimes(1);
    });
    const call = save.mock.calls[0]?.[0] as {
      agent_settings_diff?: Record<string, unknown>;
    };
    expect(call.agent_settings_diff).toEqual({ agent_kind: "openhands" });
  });

  it("persists an ACP diff matching the registry when Claude Code is selected", async () => {
    const save = vi.spyOn(SettingsService, "saveSettings");
    const { onNext } = renderStep("claude-code");
    const user = userEvent.setup();

    await user.click(screen.getByTestId("onboarding-agent-next"));

    await waitFor(() => {
      expect(save).toHaveBeenCalledTimes(1);
      expect(onNext).toHaveBeenCalledTimes(1);
    });
    const call = save.mock.calls[0]?.[0] as {
      agent_settings_diff?: Record<string, unknown>;
    };
    expect(call.agent_settings_diff).toEqual({
      agent_kind: "acp",
      acp_server: "claude-code",
      // Default-command path: the backend resolves the command from its
      // own registry, so we don't pin a stale command here.
      acp_command: [],
      // ``acp_args: []`` is reset on every save so an API-set
      // ``acp_args`` can't survive and concatenate onto the spawn
      // command at conversation-create time.
      acp_args: [],
      acp_model: "opus[1m]",
    });
  });

  it.each([
    ["codex", "codex"],
    ["gemini-cli", "gemini-cli"],
  ])("persists acp_server=%s for the matching tile", async (id, expected) => {
    const save = vi.spyOn(SettingsService, "saveSettings");
    renderStep(id as OnboardingAgentId);
    const user = userEvent.setup();

    await user.click(screen.getByTestId("onboarding-agent-next"));

    await waitFor(() => {
      expect(save).toHaveBeenCalledTimes(1);
    });
    const call = save.mock.calls[0]?.[0] as {
      agent_settings_diff?: Record<string, unknown>;
    };
    expect(
      (call.agent_settings_diff as Record<string, unknown>).acp_server,
    ).toBe(expected);
    // Canvas preselects the *preferred* default model — the registry default
    // for Codex, but the Vertex-safe override (gemini-2.5-pro) for Gemini:
    // gemini-cli re-resolves flash ids to its own default, which 404s on many
    // Vertex projects (software-agent-sdk#3532).
    expect(
      (call.agent_settings_diff as Record<string, unknown>).acp_model,
    ).toBe(getAcpPreferredDefaultModel(expected));
  });

  it("rebuilds the diff cleanly when the user flips between ACP providers", async () => {
    // Bot-flagged: would a stale claude-code selection leak into a
    // subsequent codex save? The helper rebuilds from the current
    // selectedAgentId on each Next click, so no — but pin it.
    const save = vi.spyOn(SettingsService, "saveSettings");
    const { rerender } = render(
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <ChooseAgentStep
          selectedAgentId="claude-code"
          onSelect={vi.fn()}
          onNext={vi.fn()}
        />
      </QueryClientProvider>,
    );
    const user = userEvent.setup();
    await user.click(screen.getByTestId("onboarding-agent-next"));
    await waitFor(() => {
      expect(save).toHaveBeenCalledTimes(1);
    });
    expect(
      (
        save.mock.calls[0]?.[0] as {
          agent_settings_diff?: Record<string, unknown>;
        }
      ).agent_settings_diff?.acp_server,
    ).toBe("claude-code");

    // Switch the *parent's* selection (the modal's setSelectedAgentId)
    // and click Next again — the second save should carry codex, not
    // a stale claude-code value.
    save.mockClear();
    rerender(
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <ChooseAgentStep
          selectedAgentId="codex"
          onSelect={vi.fn()}
          onNext={vi.fn()}
        />
      </QueryClientProvider>,
    );
    await user.click(screen.getByTestId("onboarding-agent-next"));
    await waitFor(() => {
      expect(save).toHaveBeenCalledTimes(1);
    });
    expect(
      (
        save.mock.calls[0]?.[0] as {
          agent_settings_diff?: Record<string, unknown>;
        }
      ).agent_settings_diff?.acp_server,
    ).toBe("codex");
  });

  it("does not advance when the save mutation fails", async () => {
    vi.spyOn(SettingsService, "saveSettings").mockRejectedValueOnce(
      new Error("boom"),
    );
    const { onNext } = renderStep("claude-code");
    const user = userEvent.setup();

    await user.click(screen.getByTestId("onboarding-agent-next"));

    // saveSettings rejects → onSuccess is not called → onNext stays untouched.
    await waitFor(
      () => {
        expect(SettingsService.saveSettings).toHaveBeenCalledTimes(1);
      },
      { timeout: 1000 },
    );
    expect(onNext).not.toHaveBeenCalled();
  });
});
