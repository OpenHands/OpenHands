import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FeedbackButton } from "./feedback-button";
import {
  configureTelemetry,
  submitPrivateFeedback,
} from "#/services/telemetry";

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
  it("requires a valid email before enabling feedback submission", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    render(<FeedbackButton />);
    fireEvent.click(screen.getByRole("button", { name: "FEEDBACK$TITLE" }));
    const info = screen.getByRole("button", { name: "FEEDBACK$CONSENT" });
    fireEvent.focus(info);
    expect(await screen.findByRole("tooltip")).toHaveTextContent(
      "FEEDBACK$CONSENT",
    );
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    const email = screen.getByLabelText("FEEDBACK$EMAIL");
    const send = screen.getByRole("button", { name: "FEEDBACK$SEND" });
    expect(email).toBeRequired();
    fireEvent.change(screen.getByLabelText("FEEDBACK$QUESTION"), {
      target: { value: "Make this easier" },
    });
    expect(send).toBeDisabled();
    fireEvent.change(email, { target: { value: "   " } });
    expect(send).toBeDisabled();
    fireEvent.change(email, { target: { value: "not-an-email" } });
    expect(send).toBeDisabled();
    fireEvent.change(email, { target: { value: "person@example.com" } });
    expect(send).toBeEnabled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("does not request a survey or capture feedback without an email", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    await expect(submitPrivateFeedback("A draft", "")).rejects.toThrow();
    await expect(submitPrivateFeedback("A draft", "   ")).rejects.toThrow();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("submits the required email with feedback without creating a profile", async () => {
    configureTelemetry({ apiKey: "phc_feedback_project" });
    localStorage.setItem("openhands-telemetry-consent", "denied");
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
    render(<FeedbackButton />);
    fireEvent.click(screen.getByRole("button", { name: "FEEDBACK$TITLE" }));
    fireEvent.change(screen.getByLabelText("FEEDBACK$QUESTION"), {
      target: { value: "Please follow up" },
    });
    fireEvent.change(screen.getByLabelText("FEEDBACK$EMAIL"), {
      target: { value: "person@example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: "FEEDBACK$SEND" }));
    expect(await screen.findByRole("status")).toHaveTextContent(
      "FEEDBACK$SUCCESS",
    );
    expect(fetchMock.mock.calls[0][0]).toContain("token=phc_feedback_project");
    expect(fetchMock.mock.calls[1][0]).toBe("https://z.openhands.dev/capture/");
    const body = JSON.parse(fetchMock.mock.calls[1][1].body);
    expect(body.api_key).toBe("phc_feedback_project");
    expect(body.properties.$survey_response).toBe("Please follow up");
    expect(body.properties.$survey_id).toBe(
      "01a19e14-c300-0000-d661-ad6bb5304fbe",
    );
    expect(body.properties.$survey_questions).toEqual([
      { id: "question-1", question: "Your feedback?" },
    ]);
    expect(body.properties.$current_url).toBeUndefined();
    expect(body.properties.feedback_email).toBe("person@example.com");
    expect(body.properties.$process_person_profile).toBe(false);
    expect(body.distinct_id).not.toBe("person@example.com");
    expect(localStorage.getItem("openhands-telemetry-consent")).toBe("denied");
    fireEvent.click(screen.getByRole("button", { name: "BUTTON$CLOSE" }));
    fireEvent.click(screen.getByRole("button", { name: "FEEDBACK$TITLE" }));
    expect(screen.getByLabelText("FEEDBACK$EMAIL")).toHaveValue("");
  });

  it("rejects a successful HTTP response without a capture acknowledgment", async () => {
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
        .mockResolvedValueOnce({ ok: true, json: async () => null }),
    );
    render(<FeedbackButton />);
    fireEvent.click(screen.getByRole("button", { name: "FEEDBACK$TITLE" }));
    const input = screen.getByLabelText("FEEDBACK$QUESTION");
    fireEvent.change(input, { target: { value: "A draft" } });
    fireEvent.change(screen.getByLabelText("FEEDBACK$EMAIL"), {
      target: { value: "person@example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: "FEEDBACK$SEND" }));
    await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
    expect(input).toHaveValue("A draft");
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("preserves the draft and reuses the survey on a capture retry", async () => {
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
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: "Ok" }),
      });
    vi.stubGlobal("fetch", fetchMock);
    render(<FeedbackButton />);
    fireEvent.click(screen.getByRole("button", { name: "FEEDBACK$TITLE" }));
    const input = screen.getByLabelText("FEEDBACK$QUESTION");
    const email = screen.getByLabelText("FEEDBACK$EMAIL");
    fireEvent.change(input, { target: { value: "A draft" } });
    fireEvent.change(email, { target: { value: "person@example.com" } });
    fireEvent.click(screen.getByRole("button", { name: "FEEDBACK$SEND" }));
    await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
    expect(input).toHaveValue("A draft");
    expect(email).toHaveValue("person@example.com");

    fireEvent.click(screen.getByRole("button", { name: "FEEDBACK$SEND" }));
    expect(await screen.findByRole("status")).toHaveTextContent(
      "FEEDBACK$SUCCESS",
    );
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(fetchMock.mock.calls[1][0]).toBe(fetchMock.mock.calls[2][0]);
  });

  it("retries a failed survey lookup rather than caching its failure", async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new Error("offline"))
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
    render(<FeedbackButton />);
    fireEvent.click(screen.getByRole("button", { name: "FEEDBACK$TITLE" }));
    fireEvent.change(screen.getByLabelText("FEEDBACK$QUESTION"), {
      target: { value: "Try again" },
    });
    fireEvent.change(screen.getByLabelText("FEEDBACK$EMAIL"), {
      target: { value: "person@example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: "FEEDBACK$SEND" }));
    expect(await screen.findByRole("alert")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "FEEDBACK$SEND" }));
    expect(await screen.findByRole("status")).toHaveTextContent(
      "FEEDBACK$SUCCESS",
    );
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(fetchMock.mock.calls[0][0]).toBe(fetchMock.mock.calls[1][0]);
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
    fireEvent.change(screen.getByLabelText("FEEDBACK$EMAIL"), {
      target: { value: "person@example.com" },
    });
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
    fireEvent.change(screen.getByLabelText("FEEDBACK$EMAIL"), {
      target: { value: "person@example.com" },
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
    fireEvent.change(screen.getByLabelText("FEEDBACK$EMAIL"), {
      target: { value: "person@example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: "FEEDBACK$SEND" }));
    expect(await screen.findByRole("alert")).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
