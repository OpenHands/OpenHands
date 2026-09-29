import React from "react";
import {
  DEFAULT_TRANSCRIPTION_MODEL,
  readTranscriptionEndpoint,
  type TranscriptionEndpoint,
} from "#/utils/transcription-endpoint-storage";

export type DictationStatus =
  | "idle"
  | "starting"
  | "recording"
  | "transcribing";

// Web Speech API types are not in lib.dom; this is the subset we use.
interface SpeechRecognitionLike {
  continuous: boolean;
  interimResults: boolean;
  onresult:
    | ((event: {
        results: ArrayLike<ArrayLike<{ transcript: string }>>;
      }) => void)
    | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
}

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

interface DictationSession {
  stop: () => void;
  cancel: () => void;
}

function getSpeechRecognition(): SpeechRecognitionConstructor | undefined {
  if (typeof window === "undefined") return undefined;
  const speechWindow = window as Window & {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  };
  return speechWindow.SpeechRecognition ?? speechWindow.webkitSpeechRecognition;
}

const canRecordAudio = () =>
  typeof MediaRecorder !== "undefined" &&
  !!navigator.mediaDevices?.getUserMedia;

async function transcribe(
  audio: Blob,
  endpoint: TranscriptionEndpoint,
): Promise<string> {
  // Whisper servers infer the codec from the file extension.
  const extension = audio.type.split(/[/;]/)[1] || "webm";
  const body = new FormData();
  body.append("file", audio, `dictation.${extension}`);
  body.append("model", endpoint.model || DEFAULT_TRANSCRIPTION_MODEL);

  const response = await fetch(
    `${endpoint.baseUrl.replace(/\/+$/, "")}/audio/transcriptions`,
    {
      method: "POST",
      headers: endpoint.apiKey
        ? { Authorization: `Bearer ${endpoint.apiKey}` }
        : undefined,
      body,
    },
  );
  if (!response.ok) {
    throw new Error(`Transcription failed with status ${response.status}`);
  }
  const { text } = await response.json();
  return typeof text === "string" ? text.trim() : "";
}

interface UseDictationOptions {
  onTranscript: (text: string) => void;
  onError: (error: unknown) => void;
}

/**
 * Push-to-toggle dictation. Uses the configured OpenAI-compatible endpoint
 * when set, otherwise the browser's speech recognition.
 */
export function useDictation({ onTranscript, onError }: UseDictationOptions) {
  const [status, setStatus] = React.useState<DictationStatus>("idle");
  const sessionRef = React.useRef<DictationSession | null>(null);
  const mountedRef = React.useRef(true);
  const endpoint = readTranscriptionEndpoint();
  const isSupported = endpoint.baseUrl
    ? canRecordAudio()
    : !!getSpeechRecognition();

  React.useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      sessionRef.current?.cancel();
    };
  }, []);

  const startSpeechRecognition = (
    Recognition: SpeechRecognitionConstructor,
  ) => {
    const recognition = new Recognition();
    recognition.continuous = true;
    recognition.interimResults = false;

    let transcript = "";
    let cancelled = false;
    recognition.onresult = (event) => {
      transcript = Array.from(event.results, (result) => result[0].transcript)
        .join(" ")
        .trim();
    };
    recognition.onerror = (event) => {
      if (event.error !== "aborted") onError(new Error(event.error));
    };
    recognition.onend = () => {
      sessionRef.current = null;
      setStatus("idle");
      if (!cancelled && transcript) onTranscript(transcript);
    };

    recognition.start();
    sessionRef.current = {
      stop: () => {
        setStatus("transcribing");
        recognition.stop();
      },
      cancel: () => {
        cancelled = true;
        recognition.abort();
      },
    };
    setStatus("recording");
  };

  const startRecording = async (target: TranscriptionEndpoint) => {
    // Enter "starting" before the permission prompt so further clicks are
    // ignored instead of opening a second, untracked stream.
    setStatus("starting");
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    if (!mountedRef.current) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    const recorder = new MediaRecorder(stream);
    const chunks: Blob[] = [];
    let cancelled = false;

    recorder.ondataavailable = (event) => chunks.push(event.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach((track) => track.stop());
      sessionRef.current = null;
      if (cancelled) return;

      setStatus("transcribing");
      try {
        const text = await transcribe(
          new Blob(chunks, { type: recorder.mimeType }),
          target,
        );
        if (text) onTranscript(text);
      } catch (error) {
        onError(error);
      } finally {
        setStatus("idle");
      }
    };

    recorder.start();
    sessionRef.current = {
      stop: () => recorder.stop(),
      cancel: () => {
        cancelled = true;
        recorder.stop();
      },
    };
    setStatus("recording");
  };

  const toggle = async () => {
    if (sessionRef.current) {
      sessionRef.current.stop();
      return;
    }
    if (status !== "idle") return;

    try {
      const Recognition = getSpeechRecognition();
      if (endpoint.baseUrl) {
        await startRecording(endpoint);
      } else if (Recognition) {
        startSpeechRecognition(Recognition);
      }
    } catch (error) {
      setStatus("idle");
      onError(error);
    }
  };

  return { status, isSupported, toggle };
}
