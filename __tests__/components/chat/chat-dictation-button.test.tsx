import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ChatDictationButton } from "#/components/features/chat/chat-dictation-button";
import { I18nKey } from "#/i18n/declaration";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { writeTranscriptionEndpoint } from "#/utils/transcription-endpoint-storage";

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
}));

class FakeSpeechRecognition {
  static instance: FakeSpeechRecognition;

  continuous = false;

  interimResults = true;

  onresult: ((event: unknown) => void) | null = null;

  onerror: ((event: { error: string }) => void) | null = null;

  onend: (() => void) | null = null;

  constructor() {
    FakeSpeechRecognition.instance = this;
  }

  start = vi.fn();

  stop = vi.fn(() => this.onend?.());

  abort = vi.fn();
}

class FakeMediaRecorder {
  mimeType = "audio/webm";

  ondataavailable: ((event: { data: Blob }) => void) | null = null;

  onstop: (() => void) | null = null;

  start = vi.fn();

  stop = vi.fn(() => {
    this.ondataavailable?.({ data: new Blob(["audio"]) });
    this.onstop?.();
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  localStorage.clear();
});

describe("ChatDictationButton", () => {
  it("renders nothing when the browser cannot dictate", () => {
    render(<ChatDictationButton onTranscript={vi.fn()} />);

    expect(screen.queryByTestId("chat-dictation-button")).toBeNull();
  });

  it("dictates with browser speech recognition until stopped", async () => {
    vi.stubGlobal("webkitSpeechRecognition", FakeSpeechRecognition);
    const user = userEvent.setup();
    const onTranscript = vi.fn();
    render(<ChatDictationButton onTranscript={onTranscript} />);

    await user.click(screen.getByTestId("chat-dictation-button"));
    FakeSpeechRecognition.instance.onresult?.({
      results: [[{ transcript: "fix the" }], [{ transcript: "login bug" }]],
    });
    const button = screen.getByTestId("chat-dictation-button");
    expect(button).toHaveAttribute(
      "aria-label",
      I18nKey.CHAT_INTERFACE$STOP_DICTATION,
    );
    await user.click(button);

    expect(onTranscript).toHaveBeenCalledWith("fix the login bug");
    expect(button).toHaveAttribute(
      "aria-label",
      I18nKey.CHAT_INTERFACE$START_DICTATION,
    );
  });

  it("transcribes recorded audio with the configured endpoint", async () => {
    writeTranscriptionEndpoint({
      baseUrl: " http://localhost:9000/v1/ ",
      apiKey: "secret",
    });
    vi.stubGlobal("MediaRecorder", FakeMediaRecorder);
    vi.stubGlobal("navigator", {
      mediaDevices: {
        getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [] }),
      },
    });
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(Response.json({ text: " hello world " }));
    const user = userEvent.setup();
    const onTranscript = vi.fn();
    render(<ChatDictationButton onTranscript={onTranscript} />);

    await user.click(screen.getByTestId("chat-dictation-button"));
    await user.click(screen.getByTestId("chat-dictation-button"));

    await waitFor(() =>
      expect(onTranscript).toHaveBeenCalledWith("hello world"),
    );
    const [url, init] = fetchSpy.mock.calls[0];
    expect(url).toBe("http://localhost:9000/v1/audio/transcriptions");
    expect(init?.headers).toEqual({ Authorization: "Bearer secret" });
    expect((init?.body as FormData).get("model")).toBe("whisper-1");
  });

  it("shows an error toast when microphone access is denied", async () => {
    writeTranscriptionEndpoint({ baseUrl: "http://localhost:9000/v1" });
    vi.stubGlobal("MediaRecorder", FakeMediaRecorder);
    vi.stubGlobal("navigator", {
      mediaDevices: {
        getUserMedia: vi.fn().mockRejectedValue(new Error("NotAllowedError")),
      },
    });
    const user = userEvent.setup();
    render(<ChatDictationButton onTranscript={vi.fn()} />);

    await user.click(screen.getByTestId("chat-dictation-button"));

    expect(displayErrorToast).toHaveBeenCalledWith(
      I18nKey.CHAT_INTERFACE$DICTATION_FAILED,
    );
  });

  it("ignores clicks while the permission prompt is open", async () => {
    writeTranscriptionEndpoint({ baseUrl: "http://localhost:9000/v1" });
    vi.stubGlobal("MediaRecorder", FakeMediaRecorder);
    let grant: (stream: MediaStream) => void = () => {};
    const getUserMedia = vi.fn(
      () =>
        new Promise<MediaStream>((resolve) => {
          grant = resolve;
        }),
    );
    vi.stubGlobal("navigator", { mediaDevices: { getUserMedia } });
    const user = userEvent.setup();
    render(<ChatDictationButton onTranscript={vi.fn()} />);

    await user.click(screen.getByTestId("chat-dictation-button"));
    await user.click(screen.getByTestId("chat-dictation-button"));

    expect(getUserMedia).toHaveBeenCalledOnce();
    await act(async () =>
      grant({ getTracks: () => [] } as unknown as MediaStream),
    );
    expect(screen.getByTestId("chat-dictation-button")).toHaveAttribute(
      "aria-pressed",
      "true",
    );
  });

  it("can still stop a recording after the input turns read-only", async () => {
    vi.stubGlobal("webkitSpeechRecognition", FakeSpeechRecognition);
    const user = userEvent.setup();
    const { rerender } = render(<ChatDictationButton onTranscript={vi.fn()} />);

    await user.click(screen.getByTestId("chat-dictation-button"));
    rerender(<ChatDictationButton onTranscript={vi.fn()} disabled />);

    expect(screen.getByTestId("chat-dictation-button")).toBeEnabled();
    await user.click(screen.getByTestId("chat-dictation-button"));
    expect(FakeSpeechRecognition.instance.stop).toHaveBeenCalledOnce();
    expect(screen.getByTestId("chat-dictation-button")).toBeDisabled();
  });

  it("releases the microphone when unmounted during the permission prompt", async () => {
    writeTranscriptionEndpoint({ baseUrl: "http://localhost:9000/v1" });
    const MediaRecorderSpy = vi.fn();
    vi.stubGlobal("MediaRecorder", MediaRecorderSpy);
    let grant: (stream: MediaStream) => void = () => {};
    vi.stubGlobal("navigator", {
      mediaDevices: {
        getUserMedia: () =>
          new Promise<MediaStream>((resolve) => {
            grant = resolve;
          }),
      },
    });
    const stopTrack = vi.fn();
    const user = userEvent.setup();
    const { unmount } = render(<ChatDictationButton onTranscript={vi.fn()} />);

    await user.click(screen.getByTestId("chat-dictation-button"));
    unmount();
    await act(async () =>
      grant({
        getTracks: () => [{ stop: stopTrack }],
      } as unknown as MediaStream),
    );

    expect(stopTrack).toHaveBeenCalledOnce();
    expect(MediaRecorderSpy).not.toHaveBeenCalled();
  });
});
