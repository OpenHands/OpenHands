import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";

import { RunPhase } from "#/components/features/automations/detail/run-phase";
import { I18nKey } from "#/i18n/declaration";
// Source of truth for translated values — not a hand-maintained duplicate.
import translationData from "#/i18n/translation.json";

type TranslationEntry = Record<string, string>;
const TRANSLATIONS = translationData as unknown as Record<
  string,
  TranslationEntry
>;

// `t()` is mocked to resolve against the real translation.json content for
// French ("fr"), the same pattern used elsewhere in this repo (see
// server-status.test.tsx) to assert on genuine translated copy rather than
// the ambient test i18n backend, which never resolves real values.
vi.mock("react-i18next", async () => {
  const actual = await vi.importActual("react-i18next");
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string) => TRANSLATIONS[key]?.fr ?? key,
      i18n: { language: "fr" },
    }),
  };
});

describe("RunPhase — known code, non-English language", () => {
  it("shows the French translation.json value for a known phase code, not the raw code", () => {
    render(<RunPhase code="sandbox_provisioning" label={null} />);

    const expected =
      TRANSLATIONS[I18nKey.AUTOMATIONS$DETAIL$PHASE_SANDBOX_PROVISIONING].fr;
    expect(screen.getByText(expected)).toBeInTheDocument();
    expect(screen.queryByText("sandbox_provisioning")).not.toBeInTheDocument();
  });
});

describe("RunPhase — unknown code (custom automations)", () => {
  it("shows the label as-is, including emoji and non-Latin text, for an unknown code", () => {
    render(<RunPhase code="poll_prs" label="🔍 Опрашиваем PR-ы" />);

    expect(screen.getByText("🔍 Опрашиваем PR-ы")).toBeInTheDocument();
  });

  it("renders nothing when the code is unknown and the label is an empty string", () => {
    render(<RunPhase code="poll_prs" label="" />);

    expect(screen.queryByTestId("run-phase")).not.toBeInTheDocument();
  });

  it("renders nothing when the code is unknown and the label is null", () => {
    render(<RunPhase code="poll_prs" label={null} />);

    expect(screen.queryByTestId("run-phase")).not.toBeInTheDocument();
  });

  it("boundary: a 200-character label (the contract's max) is kept intact in the DOM and marked for CSS truncation, not sliced in JS", () => {
    // Arrange: the longest label the backend contract allows.
    const label = "x".repeat(200);

    // Act
    render(<RunPhase code="poll_prs" label={label} />);

    // Assert: the full string is still there — truncation must be visual
    // (CSS), never a JS slice that would lose characters.
    const node = screen.getByTestId("run-phase");
    expect(node).toHaveTextContent(label);
    // Assert: the node itself carries both halves of what CSS truncation
    // needs — `truncate` alone is inert without a width to clip against.
    expect(node.className).toMatch(/\btruncate\b/);
    expect(node.className).toMatch(/max-w-/);
  });
});

describe("RunPhase — fields absent entirely (older automation service)", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders nothing and never touches the console when code/label are undefined", () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    expect(() =>
      render(<RunPhase code={undefined} label={undefined} />),
    ).not.toThrow();

    expect(screen.queryByTestId("run-phase")).not.toBeInTheDocument();
    expect(errorSpy).not.toHaveBeenCalled();
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it("renders nothing when both code and label are null (nothing has reported a phase yet)", () => {
    render(<RunPhase code={null} label={null} />);

    expect(screen.queryByTestId("run-phase")).not.toBeInTheDocument();
  });

  it("renders the label when only a label was reported and no code", () => {
    // The service accepts a phase carrying just a label, so a custom
    // automation may report one. An absent code is the most unknown code
    // there is, and the idea says an unknown code falls back to its label.
    render(<RunPhase code={null} label="Reticulating splines" />);

    expect(screen.getByTestId("run-phase")).toHaveTextContent(
      "Reticulating splines",
    );
  });
});
