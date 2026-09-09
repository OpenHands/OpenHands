export interface MeetilyUtterance {
  speaker: string;
  time: string;
  text: string;
}

export interface MeetilyActionItem {
  title: string;
  card_type: string;
  text: string;
}

export interface MeetilyDuplicate {
  title: string;
  card_type: string;
  existing_card_id: string;
  existing_title: string;
  score: number;
}

export interface MeetilyPreview {
  format: string;
  utterances: MeetilyUtterance[];
  summary: string;
  items: MeetilyActionItem[];
  source: string;
  duplicates?: MeetilyDuplicate[];
}

export interface MeetilyIngestResult extends MeetilyPreview {
  created: Array<
    Record<string, unknown> & { id: string; title: string; card_type: string }
  >;
  duplicates: MeetilyDuplicate[];
}

export interface MeetilyTranscriptPayload {
  text: string;
  format?: string;
  board_id?: string;
  channel_id?: string;
  channel_ref?: string;
  thread_ref?: string;
}
