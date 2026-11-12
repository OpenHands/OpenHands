import { render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ProjectsPage from "#/routes/projects";
import ProjectsService, {
  type Project,
  type ProjectConversation,
} from "#/api/projects-service/projects-service.api";
import { experimentalFeaturesService } from "#/api/experimental-features-service/experimental-features-service.api";
import { I18nKey } from "#/i18n/declaration";

const NOW = "2026-11-09T10:00:00Z";

const project = (id: string, name: string): Project => ({
  id,
  name,
  description: "",
  coordinator_conversation_id: null,
  worker_agent_profile_id: null,
  created_at: NOW,
  updated_at: NOW,
});

const member = (
  attention: ProjectConversation["attention"],
): ProjectConversation => ({
  conversation_id: crypto.randomUUID(),
  title: attention,
  execution_status: "finished",
  attention,
  resolved: null,
  turn: null,
  ask: null,
  automation_id: null,
  parent_conversation_id: null,
  created_at: NOW,
  updated_at: NOW,
});

function renderPage() {
  return render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <MemoryRouter initialEntries={["/projects"]}>
        <ProjectsPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("ProjectsPage", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(
      experimentalFeaturesService,
      "listExperimentalFeatures",
    ).mockResolvedValue([
      { key: "projects", label: "Projects", description: "", enabled: true },
    ] as Awaited<
      ReturnType<typeof experimentalFeaturesService.listExperimentalFeatures>
    >);
  });

  it("says which project needs you, counting its coordinator", async () => {
    vi.spyOn(ProjectsService, "listProjects").mockResolvedValue([
      project("aa000000000040008000000000000001", "Quiet"),
      project("bb000000000040008000000000000001", "Busy"),
    ]);
    vi.spyOn(ProjectsService, "listConversations").mockImplementation(
      async (projectId) =>
        projectId.startsWith("bb")
          ? {
              project_id: projectId,
              coordinator: member("needs_you"),
              conversations: [
                member("needs_input"),
                member("working"),
                member("resolved"),
              ],
            }
          : {
              project_id: projectId,
              coordinator: member("idle"),
              conversations: [member("working")],
            },
    );
    renderPage();

    const rows = await screen.findAllByTestId("project-row");
    const busy = rows.find((row) => within(row).queryByText("Busy"))!;
    const quiet = rows.find((row) => within(row).queryByText("Quiet"))!;

    expect(
      await within(busy).findByTestId("project-row-needs-you"),
    ).toHaveTextContent(`${I18nKey.PROJECTS$GROUP_NEEDS_YOU} · 2`);
    expect(
      within(quiet).queryByTestId("project-row-needs-you"),
    ).not.toBeInTheDocument();
  });
});
