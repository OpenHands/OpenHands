import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ProjectFiles } from "#/components/features/projects/project-files";
import { ProjectFilePicker } from "#/components/features/projects/project-file-picker";
import ProjectsService, {
  type ProjectArtifact,
} from "#/api/projects-service/projects-service.api";
import { experimentalFeaturesService } from "#/api/experimental-features-service/experimental-features-service.api";
import * as utils from "#/utils/utils";
import {
  artifactMediaType,
  projectArtifactKind,
} from "#/utils/project-artifact-kind";

const handleUpload = vi.fn();
vi.mock("#/hooks/chat/use-chat-attachment-upload", () => ({
  useChatAttachmentUpload: () => ({ handleUpload }),
}));

const PROJECT_ID = "aa000000000040008000000000000001";

const artifact = (
  name: string,
  updated_at: string,
  extra: Partial<ProjectArtifact> = {},
): ProjectArtifact => ({
  name,
  size: 10,
  source: "user",
  conversation_id: null,
  created_at: updated_at,
  updated_at,
  ...extra,
});

const ARTIFACTS = [
  artifact("notes.md", "2026-11-16T10:00:00Z"),
  artifact("chart.png", "2026-11-16T11:00:00Z", {
    source: "agent",
    conversation_id: "bb000000000040008000000000000001",
  }),
  artifact("spec.pdf", "2026-11-16T09:00:00Z"),
  artifact("page.html", "2026-11-16T08:00:00Z"),
  artifact("data.bin", "2026-11-16T07:00:00Z"),
];

function wrap(ui: React.ReactElement) {
  return render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <MemoryRouter>{ui}</MemoryRouter>
    </QueryClientProvider>,
  );
}

const row = async (name: string) =>
  (await screen.findAllByTestId("project-file")).find((li) =>
    li.textContent?.includes(name),
  )!;

describe("ProjectFiles", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    handleUpload.mockReset();
    vi.spyOn(
      experimentalFeaturesService,
      "listExperimentalFeatures",
    ).mockResolvedValue([
      { key: "projects", label: "Projects", description: "", enabled: true },
    ] as Awaited<
      ReturnType<typeof experimentalFeaturesService.listExperimentalFeatures>
    >);
    vi.spyOn(ProjectsService, "listArtifacts").mockResolvedValue(ARTIFACTS);
    vi.spyOn(ProjectsService, "downloadArtifact").mockImplementation(
      async (_projectId, name) =>
        new Blob(["bytes"], { type: artifactMediaType(name) }),
    );
    URL.createObjectURL = vi.fn(() => "blob:preview");
    URL.revokeObjectURL = vi.fn();
  });

  it("downloads a file under its own name", async () => {
    const save = vi.spyOn(utils, "downloadBlob").mockImplementation(() => {});
    wrap(<ProjectFiles projectId={PROJECT_ID} titles={new Map()} />);

    await userEvent.click(
      within(await row("spec.pdf")).getByTestId("project-file-download"),
    );

    await waitFor(() => expect(save).toHaveBeenCalled());
    expect(ProjectsService.downloadArtifact).toHaveBeenCalledWith(
      PROJECT_ID,
      "spec.pdf",
    );
    expect(save.mock.calls[0][1]).toBe("spec.pdf");
  });

  it("deletes a file only after the user confirms", async () => {
    const remove = vi
      .spyOn(ProjectsService, "deleteArtifact")
      .mockResolvedValue(undefined);
    wrap(<ProjectFiles projectId={PROJECT_ID} titles={new Map()} />);

    await userEvent.click(
      within(await row("notes.md")).getByTestId("project-file-delete"),
    );
    expect(remove).not.toHaveBeenCalled();
    const modal = screen.getByTestId("confirmation-modal");
    expect(modal).toHaveTextContent("PROJECTS$FILE_DELETE_CONFIRM");

    await userEvent.click(within(modal).getByTestId("confirm-button"));
    await waitFor(() =>
      expect(remove).toHaveBeenCalledWith(PROJECT_ID, "notes.md"),
    );
    await waitFor(() =>
      expect(screen.queryByTestId("confirmation-modal")).toBeNull(),
    );
  });

  it("previews an image in an img and a PDF in a frame", async () => {
    wrap(<ProjectFiles projectId={PROJECT_ID} titles={new Map()} />);

    await userEvent.click(
      within(await row("chart.png")).getByText("chart.png"),
    );
    const image = await screen.findByRole("img", { name: "chart.png" });
    expect(image).toHaveAttribute("src", "blob:preview");

    await userEvent.click(within(await row("spec.pdf")).getByText("spec.pdf"));
    const frame = await screen.findByTitle("spec.pdf");
    expect(frame.tagName).toBe("IFRAME");
    expect(frame).toHaveAttribute("src", "blob:preview");
  });

  it("shows an HTML file as text, never rendered", async () => {
    const text = vi
      .spyOn(ProjectsService, "readArtifactText")
      .mockResolvedValue("<script>alert(1)</script>");
    wrap(<ProjectFiles projectId={PROJECT_ID} titles={new Map()} />);

    await userEvent.click(
      within(await row("page.html")).getByText("page.html"),
    );

    const preview = await screen.findByTestId("project-file-preview");
    await waitFor(() =>
      expect(preview).toHaveTextContent("<script>alert(1)</script>"),
    );
    expect(preview.tagName).toBe("PRE");
    expect(text).toHaveBeenCalledWith(PROJECT_ID, "page.html");
  });
});

describe("ProjectFilePicker", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    handleUpload.mockReset();
    vi.spyOn(
      experimentalFeaturesService,
      "listExperimentalFeatures",
    ).mockResolvedValue([
      { key: "projects", label: "Projects", description: "", enabled: true },
    ] as Awaited<
      ReturnType<typeof experimentalFeaturesService.listExperimentalFeatures>
    >);
    vi.spyOn(ProjectsService, "listArtifacts").mockResolvedValue(ARTIFACTS);
    vi.spyOn(ProjectsService, "downloadArtifact").mockImplementation(
      async (_projectId, name) =>
        new Blob(["bytes"], { type: artifactMediaType(name) }),
    );
  });

  it("adds the picked file to the message as a typed File", async () => {
    const onClose = vi.fn();
    wrap(<ProjectFilePicker projectId={PROJECT_ID} onClose={onClose} />);

    const picks = await screen.findAllByTestId("project-file-pick");
    // Newest first, like the project page.
    expect(picks[0]).toHaveTextContent("chart.png");
    await userEvent.click(picks[0]);

    await waitFor(() => expect(onClose).toHaveBeenCalled());
    const [[files]] = handleUpload.mock.calls;
    expect(files).toHaveLength(1);
    expect(files[0].name).toBe("chart.png");
    expect(files[0].type).toBe("image/png");
  });
});

describe("project file kinds", () => {
  it.each([
    ["a.md", "text", "text/plain"],
    ["a.html", "text", "text/plain"],
    ["a.PNG", "image", "image/png"],
    ["a.svg", "image", "image/svg+xml"],
    ["a.pdf", "pdf", "application/pdf"],
    ["a.zip", "other", "application/octet-stream"],
    ["Makefile", "other", "application/octet-stream"],
  ])("%s is %s, typed %s", (name, kind, type) => {
    expect(projectArtifactKind(name)).toBe(kind);
    expect(artifactMediaType(name)).toBe(type);
  });
});
