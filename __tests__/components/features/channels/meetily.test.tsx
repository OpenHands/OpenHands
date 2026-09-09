import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import KanbanService from "#/api/kanban-service/kanban-service.api";
import MeetilyService from "#/api/meetily-service/meetily-service.api";
import type {
  MeetilyIngestResult,
  MeetilyPreview,
} from "#/api/meetily-service/meetily-types";
import { MeetilyImport } from "#/components/features/channels/meetily";
import { I18nKey } from "#/i18n/declaration";

const PREVIEW: MeetilyPreview = {
  format: "json",
  utterances: [{ speaker: "Ada", time: "00:01", text: "crash bug in login" }],
  summary: "crash bug in login",
  items: [
    {
      title: "crash bug in login",
      card_type: "bug",
      text: "crash bug in login",
    },
  ],
  source: "deterministic",
  duplicates: [
    {
      title: "crash bug in login",
      card_type: "bug",
      existing_card_id: "card-1",
      existing_title: "Crash bug in login form",
      score: 0.9,
    },
  ],
};

const INGEST: MeetilyIngestResult = {
  ...PREVIEW,
  created: [
    {
      id: "card-2",
      title: "Feature request: add SSO",
      card_type: "feature",
    },
  ],
  duplicates: PREVIEW.duplicates ?? [],
};

describe("MeetilyImport", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(KanbanService, "listBoards").mockResolvedValue([
      {
        id: "board-1",
        name: "Meetings",
        project_id: null,
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
      },
    ]);
  });

  it("previews detected card types and duplicate warnings", async () => {
    const user = userEvent.setup();
    const preview = vi
      .spyOn(MeetilyService, "preview")
      .mockResolvedValue(PREVIEW);

    renderWithProviders(<MeetilyImport />);
    await user.type(screen.getByTestId("meetily-paste"), "crash bug in login");
    await user.click(screen.getByTestId("meetily-preview"));

    await waitFor(() =>
      expect(preview).toHaveBeenCalledWith({
        text: "crash bug in login",
        board_id: "board-1",
      }),
    );
    expect(await screen.findByTestId("meetily-item")).toHaveTextContent("bug");
    expect(screen.getByTestId("meetily-duplicate")).toHaveTextContent(
      "Crash bug in login form",
    );
  });

  it("creates cards from a transcript and links to the board", async () => {
    const user = userEvent.setup();
    vi.spyOn(MeetilyService, "preview").mockResolvedValue(PREVIEW);
    const ingest = vi.spyOn(MeetilyService, "ingest").mockResolvedValue(INGEST);

    renderWithProviders(<MeetilyImport />);
    await user.type(screen.getByTestId("meetily-paste"), "Feature request");
    await user.click(screen.getByTestId("meetily-create"));

    await waitFor(() =>
      expect(ingest).toHaveBeenCalledWith({
        text: "Feature request",
        board_id: "board-1",
      }),
    );
    expect(await screen.findByTestId("meetily-created")).toBeInTheDocument();
    expect(screen.getByTestId("meetily-open-board")).toHaveAttribute(
      "href",
      "/kanban",
    );
  });
});

describe("meetily i18n keys", () => {
  it("exposes import copy keys", () => {
    expect(I18nKey.MEETILY$TITLE).toBe("MEETILY$TITLE");
    expect(I18nKey.MEETILY$PREVIEW).toBe("MEETILY$PREVIEW");
  });
});
