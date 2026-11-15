import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { ActionSection } from "#/components/features/automations/detail/action-section";
import type { Automation } from "#/types/automation";

vi.mock("#/hooks/query/use-automation-tarball-files", () => ({
  useAutomationTarballFiles: () => ({
    data: undefined,
    isPending: true,
    isError: false,
  }),
}));

vi.mock("#/hooks/query/use-agent-profiles", () => ({
  useAgentProfiles: () => ({
    data: {
      profiles: [
        { id: "agent-reviewer", name: "Reviewer", llm_profile_ref: "fast" },
        { id: "agent-plain", name: "Plain", llm_profile_ref: null },
        {
          id: "agent-claude",
          name: "Claude",
          agent_kind: "acp",
          llm_profile_ref: null,
        },
      ],
    },
    isLoading: false,
  }),
}));

vi.mock("#/hooks/query/use-llm-profiles", () => ({
  useLlmProfiles: () => ({ data: { active_profile: "everyday" } }),
}));

const automation: Automation = {
  id: "auto-1",
  name: "PR Review Bot",
  prompt: "Review PRs",
  trigger: { type: "event", source: "github", on: "pull_request.opened" },
  enabled: true,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  model: "review-model",
  repositories: [{ url: "acme/frontend-app", ref: "main" }],
  plugins: ["github.com/acme/review-plugin"],
};

describe("ActionSection", () => {
  it("shows the prompt with the model, repositories and plugins it runs with", () => {
    render(<ActionSection automation={automation} />);

    expect(screen.getByText("AUTOMATIONS$DETAIL$PROMPT")).toBeInTheDocument();
    expect(screen.getByTestId("automation-prompt-content")).toHaveTextContent(
      "Review PRs",
    );
    expect(screen.getByText("review-model")).toBeInTheDocument();
    expect(screen.getByText("AUTOMATIONS$DETAIL$PLUGINS")).toBeInTheDocument();
  });

  it("shows the script instead of a prompt for a script automation", () => {
    render(
      <ActionSection
        automation={{
          ...automation,
          prompt: null,
          entrypoint: "python main.py",
        }}
      />,
    );

    expect(screen.getByText("AUTOMATIONS$DETAIL$SCRIPT")).toBeInTheDocument();
    expect(
      screen.getByTestId("automation-script-entrypoint"),
    ).toHaveTextContent("python main.py");
    expect(
      screen.queryByTestId("automation-prompt-content"),
    ).not.toBeInTheDocument();
  });

  it("renders every repository as owner/repo with its branch badge", () => {
    render(
      <ActionSection
        automation={{
          ...automation,
          repositories: [
            { url: "https://github.com/qa-example/qa-repo", ref: "main" },
            { url: "git@gitlab.com:qa-example/docs.git", ref: "v2" },
            { url: "qa-example/tools" },
            {
              url: "https://bitbucket.example.com/scm/DEM/demo-app.git",
              provider: "bitbucket_data_center",
            },
            {
              url: "https://example.com/bitbucket/scm/~ALICE/notes.git",
              provider: "bitbucket_data_center",
            },
          ],
        }}
      />,
    );

    const rows = screen.getAllByTestId("automation-repository");
    expect(rows.map((row) => row.textContent)).toEqual([
      "qa-example/qa-repomain",
      "qa-example/docsv2",
      "qa-example/tools",
      "DEM/demo-app",
      "~ALICE/notes",
    ]);
  });

  it("omits repositories and plugins the automation does not have", () => {
    render(
      <ActionSection
        automation={{ ...automation, repositories: undefined, plugins: [] }}
      />,
    );

    expect(
      screen.queryByText("AUTOMATIONS$DETAIL$REPOSITORIES"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText("AUTOMATIONS$DETAIL$PLUGINS"),
    ).not.toBeInTheDocument();
  });

  it("shows the agent profile and the LLM profile it brings", () => {
    render(
      <ActionSection
        automation={{
          ...automation,
          prompt: null,
          model: null,
          agent_profile_id: "agent-reviewer",
        }}
      />,
    );

    expect(
      screen.getByTestId("automation-agent-profile-name"),
    ).toHaveTextContent("Reviewer");
    expect(
      screen.getByTestId("automation-agent-profile-llm"),
    ).toHaveTextContent("fast");
    expect(
      screen.getByText("AUTOMATIONS$DETAIL$FROM_AGENT_PROFILE"),
    ).toBeInTheDocument();
  });

  it("names the active LLM profile for an agent profile without its own", () => {
    render(
      <ActionSection
        automation={{ ...automation, agent_profile_id: "agent-plain" }}
      />,
    );

    expect(
      screen.getByTestId("automation-agent-profile-llm"),
    ).toHaveTextContent("everyday");
  });

  it("names the ACP agent, not an LLM profile, for an ACP agent profile", () => {
    render(
      <ActionSection
        automation={{ ...automation, agent_profile_id: "agent-claude" }}
      />,
    );

    expect(
      screen.getByTestId("automation-agent-profile-llm"),
    ).toHaveTextContent("CONVERSATION$ACP_AGENT_GENERIC");
  });

  it("flags a profile the library no longer has", () => {
    render(
      <ActionSection
        automation={{ ...automation, agent_profile_id: "agent-gone" }}
      />,
    );

    expect(
      screen.getByTestId("automation-agent-profile-name"),
    ).toHaveTextContent("AUTOMATIONS$UNAVAILABLE_AGENT_PROFILE");
  });
});
