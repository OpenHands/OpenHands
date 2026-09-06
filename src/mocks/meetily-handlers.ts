import { http, HttpResponse } from "msw";
import {
  MEETINGS_ACTION_PATH,
  MEETINGS_TRANSCRIPT_PATH,
} from "#/api/meetily-service/meetily-constants";
import type {
  MeetilyIngestResult,
  MeetilyPreview,
  MeetilyTranscriptPayload,
} from "#/api/meetily-service/meetily-types";

function previewFrom(payload: MeetilyTranscriptPayload): MeetilyPreview {
  const text = payload.text || "";
  return {
    format: payload.format || "markdown",
    utterances: [{ speaker: "", time: "", text }],
    summary: text.slice(0, 80),
    items: text ? [{ title: text.slice(0, 80), card_type: "task", text }] : [],
    source: "deterministic",
    duplicates: [],
  };
}

export const MEETILY_HANDLERS = [
  http.post(MEETINGS_ACTION_PATH, async ({ request }) => {
    const payload = (await request.json()) as MeetilyTranscriptPayload;
    return HttpResponse.json(previewFrom(payload));
  }),
  http.post(MEETINGS_TRANSCRIPT_PATH, async ({ request }) => {
    const payload = (await request.json()) as MeetilyTranscriptPayload;
    const preview = previewFrom(payload);
    const result: MeetilyIngestResult = {
      ...preview,
      created: payload.board_id
        ? preview.items.map((item, index) => ({
            id: `card-${index + 1}`,
            title: item.title,
            card_type: item.card_type,
          }))
        : [],
      duplicates: [],
    };
    return HttpResponse.json(result, { status: payload.board_id ? 201 : 200 });
  }),
];
