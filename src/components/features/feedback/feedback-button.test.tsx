import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FeedbackButton } from "./feedback-button";
import { configureTelemetry } from "#/services/telemetry";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

beforeEach(() => {
  vi.stubEnv("VITE_POSTHOG_SURVEY_ID", "01a19e14-c300-0000-d661-ad6bb5304fbe");
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  configureTelemetry({});
  localStorage.removeItem("openhands-telemetry-consent");
});

describe("FeedbackButton", () => {
  it("opens the form immediately and submits only the entered message", async () => {
    configureTelemetry({ apiKey: "phc_feedback_project" });
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          surveys: [
            {
              id: "01a19e14-c300-0000-d661-ad6bb5304fbe",
              questions: [{ id: "question-1", question: "Your feedback?" }],
            },
          ],
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: "Ok" }),
      });
    vi.stubGlobal("fetch", fetchMock);
    localStorage.setItem("openhands-telemetry-consent", "denied");
    render(<FeedbackButton />);
    fireEvent.click(screen.getByRole("button", { name: "FEEDBACK$TITLE" }));
    const info = screen.getByRole("button", { name: "FEEDBACK$CONSENT" });
    fireEvent.focus(info);
    expect(await screen.findByRole("tooltip")).toHaveTextContent(
      "FEEDBACK$CONSENT",
    );
    expect(screen.getByLabelText("FEEDBACK$QUESTION")).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("FEEDBACK$QUESTION"), {
      target: { value: "Make this easier" },
    });
    fireEvent.click(screen.getByRole("button", { name: "FEEDBACK$SEND" }));
    expect(await screen.findByRole("status")).toHaveTextContent(
      "FEEDBACK$SUCCESS",
    );
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[0][0]).toContain("token=phc_feedback_project");
    const [, options] = fetchMock.mock.calls[1];
    const body = JSON.parse(options.body);
    expect(body.api_key).toBe("phc_feedback_project");
    expect(body.properties.$survey_response).toBe("Make this easier");
    expect(body.properties.$survey_id).toBe(
      "01a19e14-c300-0000-d661-ad6bb5304fbe",
    );
    expect(body.properties.$current_url).toBeUndefined();
    expect(body.properties.$survey_questions).toEqual([
      { id: "question-1", question: "Your feedback?" },
    ]);
    expect(localStorage.getItem("openhands-telemetry-consent")).toBe("denied");
  });

  it("preserves the message after a failed submission", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            surveys: [
              {
                id: "01a19e14-c300-0000-d661-ad6bb5304fbe",
                questions: [{ id: "question-1", question: "Your feedback?" }],
              },
            ],
          }),
        })
        .mockRejectedValueOnce(new Error("offline")),
    );
    render(<FeedbackButton />);
    fireEvent.click(screen.getByRole("button", { name: "FEEDBACK$TITLE" }));
    const input = screen.getByLabelText("FEEDBACK$QUESTION");
    fireEvent.change(input, { target: { value: "A draft" } });
    fireEvent.click(screen.getByRole("button", { name: "FEEDBACK$SEND" }));
    await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
    expect(input).toHaveValue("A draft");
  });

  it("does not send to a project that does not contain the survey", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue({ ok: true, json: async () => ({ surveys: [] }) });
    vi.stubGlobal("fetch", fetchMock);
    render(<FeedbackButton />);
    fireEvent.click(screen.getByRole("button", { name: "FEEDBACK$TITLE" }));
    const input = screen.getByLabelText("FEEDBACK$QUESTION");
    fireEvent.change(input, { target: { value: "A draft" } });
    fireEvent.click(screen.getByRole("button", { name: "FEEDBACK$SEND" }));
    expect(await screen.findByRole("alert")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(input).toHaveValue("A draft");
  });

  it("does not send feedback when no survey is configured", async () => {
    vi.stubEnv("VITE_POSTHOG_SURVEY_ID", "");
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    render(<FeedbackButton />);
    fireEvent.click(screen.getByRole("button", { name: "FEEDBACK$TITLE" }));
    fireEvent.change(screen.getByLabelText("FEEDBACK$QUESTION"), {
      target: { value: "A draft" },
    });
    fireEvent.click(screen.getByRole("button", { name: "FEEDBACK$SEND" }));
    expect(await screen.findByRole("alert")).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("does not send feedback when the embedding host disables telemetry", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    configureTelemetry(false);
    render(<FeedbackButton />);
    fireEvent.click(screen.getByRole("button", { name: "FEEDBACK$TITLE" }));
    fireEvent.change(screen.getByLabelText("FEEDBACK$QUESTION"), {
      target: { value: "A draft" },
    });
    fireEvent.click(screen.getByRole("button", { name: "FEEDBACK$SEND" }));
    expect(await screen.findByRole("alert")).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
