import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ErrorMessage } from "./error-message";

const i18nMocks = vi.hoisted(() => ({
  exists: vi.fn(),
}));

const useTranslationMock = vi.hoisted(() =>
  vi.fn((namespace: string) => ({
    t: (key: string) =>
      namespace === "openhands" ? `translated:${key}` : `missing:${key}`,
  })),
);

vi.mock("#/i18n", () => ({
  default: {
    exists: i18nMocks.exists,
  },
}));

vi.mock("react-i18next", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-i18next")>()),
  useTranslation: useTranslationMock,
}));

interface ErrorMessageScenario {
  errorId?: string;
  translationExists?: boolean;
  defaultMessage?: string;
}

function renderChatError({
  errorId,
  translationExists = false,
  defaultMessage = "Agent failed during **repository setup**.",
}: ErrorMessageScenario = {}) {
  i18nMocks.exists.mockReturnValue(translationExists);

  return render(
    <ErrorMessage errorId={errorId} defaultMessage={defaultMessage} />,
  );
}

afterEach(() => {
  vi.clearAllMocks();
});

describe("chat error details", () => {
  it("shows the generic heading and lets the user reveal and hide details", () => {
    renderChatError();

    expect(i18nMocks.exists).not.toHaveBeenCalled();
    expect(
      screen.getByText("translated:CHAT_INTERFACE$AGENT_ERROR_MESSAGE"),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("markdown-renderer")).not.toBeInTheDocument();

    const detailsToggle = screen.getByRole("button");
    fireEvent.click(detailsToggle);

    const details = screen.getByTestId("markdown-renderer");
    expect(details).toHaveTextContent("Agent failed during repository setup.");
    expect(within(details).getByText("repository setup").tagName).toBe(
      "STRONG",
    );

    fireEvent.click(detailsToggle);

    expect(screen.queryByTestId("markdown-renderer")).not.toBeInTheDocument();
  });

  it("uses the generic heading when the backend error ID is unknown", () => {
    renderChatError({ errorId: "ERROR$NEW_BACKEND_FAILURE" });

    expect(i18nMocks.exists).toHaveBeenCalledOnce();
    expect(i18nMocks.exists).toHaveBeenCalledWith("ERROR$NEW_BACKEND_FAILURE");
    expect(
      screen.getByText("translated:CHAT_INTERFACE$AGENT_ERROR_MESSAGE"),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("ERROR$NEW_BACKEND_FAILURE"),
    ).not.toBeInTheDocument();
  });

  it("uses a translated backend error ID when it is available", () => {
    renderChatError({
      errorId: "ERROR$RATE_LIMITED",
      translationExists: true,
    });

    expect(i18nMocks.exists).toHaveBeenCalledOnce();
    expect(i18nMocks.exists).toHaveBeenCalledWith("ERROR$RATE_LIMITED");
    expect(
      screen.getByText("translated:ERROR$RATE_LIMITED"),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("translated:CHAT_INTERFACE$AGENT_ERROR_MESSAGE"),
    ).not.toBeInTheDocument();
  });
});
