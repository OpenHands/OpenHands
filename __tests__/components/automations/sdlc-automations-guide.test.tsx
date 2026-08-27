import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { I18nKey } from "#/i18n/declaration";
import {
  SdlcAutomationsGuide,
  buildSdlcPhaseStates,
} from "#/components/features/automations/sdlc-automations-guide";
import type { Automation } from "#/types/automation";
import type { MCPServerConfig } from "#/types/mcp-server";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) => {
      if (values?.progress !== undefined) {
        return `${key}:${String(values.progress)}`;
      }
      if (values?.phase !== undefined) {
        return `${key}:${String(values.phase)}`;
      }
      if (values?.integrations !== undefined) {
        return `${key}:${String(values.integrations)}`;
      }
      if (values?.count !== undefined) {
        return `${key}:${String(values.count)}`;
      }
      return key;
    },
  }),
}));

const githubServer: MCPServerConfig = {
  id: "github",
  type: "shttp",
  url: "https://api.githubcopilot.com/mcp/",
};

function installedReviewer(enabled: boolean): Automation {
  return {
    id: "installed-reviewer",
    name: "GitHub Code Review Agent",
    prompt: "Review pull requests",
    trigger: { type: "event", source: "github" },
    enabled,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };
}

describe("SDLC automations guide", () => {
  it("derives phase maturity from created automations and integration readiness", () => {
    const empty = buildSdlcPhaseStates([], []);
    expect(empty.find((phase) => phase.id === "review")?.status).toBe(
      "blocked",
    );

    const ready = buildSdlcPhaseStates([], [githubServer]);
    expect(ready.find((phase) => phase.id === "review")?.status).toBe(
      "available",
    );

    const partial = buildSdlcPhaseStates(
      [installedReviewer(false)],
      [githubServer],
    );
    expect(partial.find((phase) => phase.id === "review")?.status).toBe(
      "partial",
    );

    const complete = buildSdlcPhaseStates(
      [installedReviewer(true)],
      [githubServer],
    );
    expect(complete.find((phase) => phase.id === "review")?.status).toBe(
      "complete",
    );
  });

  it("shows blockers and starts an automation from the selected phase", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();

    render(
      <SdlcAutomationsGuide
        installedAutomations={[]}
        installedServers={[]}
        onSelect={onSelect}
        onOpenInstalled={vi.fn()}
      />,
    );

    await user.click(screen.getByTestId("sdlc-phase-implement"));

    const issueToPr = screen.getByTestId("sdlc-opportunity-github-issue-to-pr");
    expect(
      within(issueToPr).getByText(
        `${I18nKey.SDLC_AUTOMATIONS_GUIDE$MISSING_INTEGRATIONS}:GitHub`,
      ),
    ).toBeInTheDocument();

    await user.click(
      within(issueToPr).getByRole("button", {
        name: I18nKey.AUTOMATIONS$CREATE_AUTOMATION_BUTTON,
      }),
    );
    expect(onSelect).toHaveBeenCalledWith(
      expect.objectContaining({ id: "github-issue-to-pr" }),
    );
  });

  it("shows progress and opens an existing automation for refinement", async () => {
    const user = userEvent.setup();
    const reviewer = installedReviewer(true);
    const onOpenInstalled = vi.fn();

    render(
      <SdlcAutomationsGuide
        installedAutomations={[reviewer]}
        installedServers={[githubServer]}
        onSelect={vi.fn()}
        onOpenInstalled={onOpenInstalled}
      />,
    );

    expect(screen.getByRole("progressbar")).toHaveAttribute(
      "aria-valuenow",
      "20",
    );
    await user.click(screen.getByTestId("sdlc-phase-review"));

    const installed = screen.getByTestId("sdlc-opportunity-github-pr-reviewer");
    expect(
      within(installed).getByText(I18nKey.AUTOMATIONS$ACTIVE),
    ).toBeInTheDocument();

    await user.click(
      within(installed).getByRole("button", {
        name: I18nKey.AUTOMATIONS$EDIT,
      }),
    );
    expect(onOpenInstalled).toHaveBeenCalledWith(reviewer);
  });
});
