import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { NewProjectModal } from "#/components/features/projects/new-project-modal";
import ProjectsService, {
  type Project,
} from "#/api/projects-service/projects-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import { DEFAULT_SETTINGS } from "#/services/settings";

vi.mock("#/api/conversation-workspace", () => ({
  resolveNewConversationWorkspace: vi.fn(async () => ({
    workingDir: "/work/new",
  })),
}));
vi.mock("#/hooks/query/use-agent-profiles", () => ({
  useAgentProfiles: () => ({ data: [] }),
}));
vi.mock("#/hooks/query/use-local-workspaces", () => ({
  useLocalWorkspaces: () => ({ data: [] }),
}));
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({ backend: null }),
}));

const PROJECT = { id: "aa", name: "Tools" } as Project;

function withConversationSettings(conversationSettings: object) {
  vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
    ...DEFAULT_SETTINGS,
    conversation_settings: conversationSettings,
  } as Awaited<ReturnType<typeof SettingsService.getSettings>>);
}

async function createProject() {
  const create = vi
    .spyOn(ProjectsService, "createProject")
    .mockResolvedValue(PROJECT);
  render(
    <QueryClientProvider client={new QueryClient()}>
      <NewProjectModal onClose={vi.fn()} onCreated={vi.fn()} />
    </QueryClientProvider>,
  );
  await userEvent.type(screen.getByTestId("new-project-name"), "Tools");
  await userEvent.click(screen.getByTestId("new-project-create"));
  await waitFor(() => expect(create).toHaveBeenCalledTimes(1));
  return create.mock.calls[0][0].coordinator;
}

describe("NewProjectModal", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("starts the coordinator with the user's approval settings", async () => {
    withConversationSettings({
      confirmation_mode: true,
      security_analyzer: null,
      max_iterations: 80,
    });

    expect(await createProject()).toEqual({
      working_dir: "/work/new",
      title: "Tools",
      confirmation_policy: { kind: "AlwaysConfirm" },
      max_iterations: 80,
    });
  });

  it("passes the risk analyzer along when the user picked it", async () => {
    withConversationSettings({
      confirmation_mode: true,
      security_analyzer: "llm",
    });

    expect(await createProject()).toMatchObject({
      confirmation_policy: { kind: "ConfirmRisky" },
      security_analyzer: { kind: "LLMSecurityAnalyzer" },
    });
  });

  it("never asks when the user turned approval off", async () => {
    withConversationSettings({ confirmation_mode: false });

    expect(await createProject()).toMatchObject({
      confirmation_policy: { kind: "NeverConfirm" },
    });
  });
});
