export const TRANSCRIPTION_ENDPOINT_STORAGE_KEY =
  "openhands-transcription-endpoint";

export const DEFAULT_TRANSCRIPTION_MODEL = "whisper-1";

/** OpenAI-compatible `/audio/transcriptions` provider, kept in this browser only. */
export interface TranscriptionEndpoint {
  baseUrl: string;
  apiKey: string;
  model: string;
}

const EMPTY_TRANSCRIPTION_ENDPOINT: TranscriptionEndpoint = {
  baseUrl: "",
  apiKey: "",
  model: "",
};

const readString = (value: unknown): string =>
  typeof value === "string" ? value.trim() : "";

export function readTranscriptionEndpoint(): TranscriptionEndpoint {
  if (typeof window === "undefined") return EMPTY_TRANSCRIPTION_ENDPOINT;

  try {
    const stored = JSON.parse(
      window.localStorage.getItem(TRANSCRIPTION_ENDPOINT_STORAGE_KEY) ?? "{}",
    );
    return {
      baseUrl: readString(stored?.baseUrl),
      apiKey: readString(stored?.apiKey),
      model: readString(stored?.model),
    };
  } catch {
    return EMPTY_TRANSCRIPTION_ENDPOINT;
  }
}

export function writeTranscriptionEndpoint(
  update: Partial<TranscriptionEndpoint>,
): void {
  try {
    window.localStorage.setItem(
      TRANSCRIPTION_ENDPOINT_STORAGE_KEY,
      JSON.stringify({ ...readTranscriptionEndpoint(), ...update }),
    );
  } catch {
    // Ignore storage failures; dictation falls back to browser recognition.
  }
}
