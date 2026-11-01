export const TRANSCRIPTION_ENDPOINT_STORAGE_KEY =
  "openhands-transcription-endpoint";

/** The API key lives in sessionStorage so it is not persisted at rest. */
export const TRANSCRIPTION_API_KEY_STORAGE_KEY =
  "openhands-transcription-api-key";

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
      apiKey: readString(
        window.sessionStorage.getItem(TRANSCRIPTION_API_KEY_STORAGE_KEY),
      ),
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
    const { apiKey, ...rest } = { ...readTranscriptionEndpoint(), ...update };
    window.localStorage.setItem(
      TRANSCRIPTION_ENDPOINT_STORAGE_KEY,
      JSON.stringify(rest),
    );
    window.sessionStorage.setItem(TRANSCRIPTION_API_KEY_STORAGE_KEY, apiKey);
  } catch {
    // Ignore storage failures; dictation falls back to browser recognition.
  }
}
