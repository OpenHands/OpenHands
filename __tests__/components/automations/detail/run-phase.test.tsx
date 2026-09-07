import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";

import {
  formatRunPhaseAge,
  resolveRunPhaseText,
  RunPhase,
} from "#/components/features/automations/detail/run-phase";
import { I18nKey } from "#/i18n/declaration";
// Source of truth for translated values — not a hand-maintained duplicate.
import translationData from "#/i18n/translation.json";
import { AutomationRunStatus } from "#/types/automation";
import { formatRelativeTime } from "#/utils/format-relative-time";

type TranslationEntry = Record<string, string>;
const TRANSLATIONS = translationData as unknown as Record<
  string,
  TranslationEntry
>;

// `t()` is mocked to resolve against the real translation.json content for
// French ("fr"), the same pattern used elsewhere in this repo (see
// server-status.test.tsx) to assert on genuine translated copy rather than
// the ambient test i18n backend, which never resolves real values.
function translate(key: string, options?: Record<string, unknown>): string {
  const value = TRANSLATIONS[key]?.fr ?? key;
  if (!options) return value;
  // Interpolate the same way i18next does, so a test can assert on "il y a
  // 12min" rather than on the untouched "il y a {{count}}min" placeholder.
  return value.replace(/{{(\w+)}}/g, (_, name: string) =>
    String(options[name] ?? `{{${name}}}`),
  );
}

vi.mock("react-i18next", async () => {
  const actual = await vi.importActual("react-i18next");
  return {
    ...actual,
    useTranslation: () => ({
      t: translate,
      i18n: { language: "fr" },
    }),
  };
});

const minutesAgo = (minutes: number) =>
  new Date(Date.now() - minutes * 60_000).toISOString();

const RUNNING = AutomationRunStatus.RUNNING;

describe("RunPhase — known code, non-English language", () => {
  it("shows the French translation.json value for a known phase code, not the raw code", () => {
    render(
      <RunPhase status={RUNNING} code="sandbox_provisioning" label={null} />,
    );

    const expected =
      TRANSLATIONS[I18nKey.AUTOMATIONS$DETAIL$PHASE_SANDBOX_PROVISIONING].fr;
    expect(screen.getByText(expected)).toBeInTheDocument();
    expect(screen.queryByText("sandbox_provisioning")).not.toBeInTheDocument();
  });
});

describe("RunPhase — unknown code (custom automations)", () => {
  it("shows the label as-is, including emoji and non-Latin text, for an unknown code", () => {
    render(
      <RunPhase status={RUNNING} code="poll_prs" label="🔍 Опрашиваем PR-ы" />,
    );

    expect(screen.getByText("🔍 Опрашиваем PR-ы")).toBeInTheDocument();
  });

  it("shows the raw code when the automation reported a code and no label", () => {
    // The service accepts `{"code": "checking_out"}` with no label at all, so
    // this run has a real phase and dropping it would show nothing.
    render(<RunPhase status={RUNNING} code="checking_out" label={null} />);

    expect(screen.getByTestId("run-phase")).toHaveTextContent("checking_out");
  });

  it.each([
    ["null code and empty label", null, ""],
    ["undefined on both fields (an older service)", undefined, undefined],
    ["null on both fields", null, null],
    ["whitespace on both fields", "   ", "   "],
  ])(
    "renders nothing and never touches the console with %s",
    (_case, code, label) => {
      const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
      const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

      expect(() =>
        render(<RunPhase status={RUNNING} code={code} label={label} />),
      ).not.toThrow();

      expect(screen.queryByTestId("run-phase")).not.toBeInTheDocument();
      expect(errorSpy).not.toHaveBeenCalled();
      expect(warnSpy).not.toHaveBeenCalled();
    },
  );

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("boundary: a 200-character label (the contract's max) reaches the DOM whole, so truncation stays visual", () => {
    // Arrange: the longest label the backend contract allows.
    const label = "x".repeat(200);

    // Act
    render(<RunPhase status={RUNNING} code="poll_prs" label={label} />);

    // Assert: the full string is still there — clipping must be CSS, never a
    // JS slice that would lose characters no tooltip could bring back.
    expect(screen.getByTestId("run-phase")).toHaveTextContent(label);
  });

  it("renders the label when only a label was reported and no code", () => {
    // The service accepts a phase carrying just a label, so a custom
    // automation may report one. An absent code is the most unknown code
    // there is, and an unknown code falls back to its label.
    render(
      <RunPhase status={RUNNING} code={null} label="Reticulating splines" />,
    );

    expect(screen.getByTestId("run-phase")).toHaveTextContent(
      "Reticulating splines",
    );
  });
});

describe("RunPhase — phase text is data, not a lookup into JavaScript itself", () => {
  // `phase_code` is author-supplied, so a plain-object lookup answers these
  // from `Object.prototype` and hands a function to `t()` — taking the card
  // tree down inside render instead of falling back to the label.
  it.each(["toString", "constructor", "__proto__", "hasOwnProperty"])(
    "treats %s as an ordinary unknown code and shows the label",
    (code) => {
      expect(() =>
        render(<RunPhase status={RUNNING} code={code} label="Полный аудит" />),
      ).not.toThrow();

      expect(screen.getByTestId("run-phase")).toHaveTextContent("Полный аудит");
    },
  );
});

describe("resolveRunPhaseText — one answer for the row and its tooltip", () => {
  // The row shows a clipped copy of this text and the tooltip shows all of
  // it, so both have to resolve the same phase the same way; a surface that
  // resolved it on its own could show one thing and reveal another.
  const t = (key: string) => TRANSLATIONS[key]?.fr ?? key;

  it("resolves a known code to its translation", () => {
    expect(resolveRunPhaseText(t, "sandbox_provisioning", null)).toBe(
      TRANSLATIONS[I18nKey.AUTOMATIONS$DETAIL$PHASE_SANDBOX_PROVISIONING].fr,
    );
  });

  it("resolves an unknown code to the author's label, verbatim", () => {
    expect(resolveRunPhaseText(t, "poll_prs", "🔍 Опрашиваем PR-ы")).toBe(
      "🔍 Опрашиваем PR-ы",
    );
  });

  it("resolves a phase carrying only a label to that label", () => {
    expect(resolveRunPhaseText(t, null, "Reticulating splines")).toBe(
      "Reticulating splines",
    );
  });

  it("resolves a phase carrying only a code to that code, verbatim", () => {
    // Mirrors the backend contract: `code` and `label` are independently
    // optional, so a code-only phase is valid and must reach the screen.
    expect(resolveRunPhaseText(t, "checking_out", null)).toBe("checking_out");
    expect(resolveRunPhaseText(t, "checking_out", "")).toBe("checking_out");
  });

  it("prefers the author's label over the raw code when both are present", () => {
    expect(resolveRunPhaseText(t, "poll_prs", "Polling PRs")).toBe(
      "Polling PRs",
    );
  });

  it("falls through a whitespace-only label, which the service stores as sent", () => {
    // Only a phase blank on *both* fields is rejected: `{"code": "x",
    // "label": "  "}` is recorded verbatim, and rendering that label would
    // put an empty span on the row with the age dangling beside it.
    expect(resolveRunPhaseText(t, "checking_out", "   ")).toBe("checking_out");
    expect(resolveRunPhaseText(t, "   ", "   ")).toBeNull();
  });

  it("resolves to null when there is nothing to show", () => {
    expect(resolveRunPhaseText(t, "", "")).toBeNull();
    expect(resolveRunPhaseText(t, null, null)).toBeNull();
  });
});

describe("formatRunPhaseAge — telling a moving run from a stalled one", () => {
  it("reports how long the run has held the phase, localized", () => {
    expect(formatRunPhaseAge(minutesAgo(12), "fr", translate)).toBe(
      "il y a 12min",
    );
  });

  it("reports a phase written seconds ago as 'just now', not as zero minutes", () => {
    expect(formatRunPhaseAge(minutesAgo(0), "fr", translate)).toBe(
      TRANSLATIONS[I18nKey.AUTOMATIONS$DETAIL$TIME_JUST_NOW].fr,
    );
  });

  it("boundary: 60 minutes rolls over to the hours wording", () => {
    expect(formatRunPhaseAge(minutesAgo(60), "fr", translate)).toBe(
      "il y a 1h",
    );
  });

  it("returns null when the service reported no timestamp", () => {
    expect(formatRunPhaseAge(null, "fr", translate)).toBeNull();
    expect(formatRunPhaseAge(undefined, "fr", translate)).toBeNull();
  });

  it("negative: returns null for timestamps the formatter would print as garbage", () => {
    // The guard is the whole point: left to itself, the shared relative-time
    // formatter prints "Invalid Date" for an unparseable string and a 1970
    // date for the epoch, which is how the backend leaves a datetime unset.
    expect(formatRelativeTime("not-a-date", "fr", translate)).toMatch(
      /Invalid/,
    );
    expect(formatRelativeTime("1970-01-01T00:00:00Z", "fr", translate)).toMatch(
      /1970/,
    );

    expect(formatRunPhaseAge("not-a-date", "fr", translate)).toBeNull();
    expect(
      formatRunPhaseAge("1970-01-01T00:00:00Z", "fr", translate),
    ).toBeNull();
  });
});

describe("RunPhase — how long the run has been in this phase", () => {
  it("shows the age beside the phase, so a stalled run reads differently from a moving one", () => {
    render(
      <RunPhase
        status={RUNNING}
        code="running_agent"
        label={null}
        updatedAt={minutesAgo(41)}
      />,
    );

    expect(screen.getByTestId("run-phase-age")).toHaveTextContent(
      "il y a 41min",
    );
  });

  it("keeps the age in its own node, outside the clipped text, so a maximum-length label cannot push it out of sight", () => {
    // Arrange: the longest label the backend contract allows, in the surface
    // with the least room.
    const label = "x".repeat(200);

    render(
      <RunPhase
        status={RUNNING}
        code="poll_prs"
        label={label}
        updatedAt={minutesAgo(7)}
      />,
    );

    const text = screen.getByTestId("run-phase");
    const age = screen.getByTestId("run-phase-age");
    expect(text).not.toContainElement(age);
  });

  it("shows no age at all against a service that reports phases without a timestamp", () => {
    render(<RunPhase status={RUNNING} code="running_agent" label={null} />);

    expect(screen.getByTestId("run-phase")).toBeInTheDocument();
    expect(screen.queryByTestId("run-phase-age")).not.toBeInTheDocument();
  });

  it.each([
    ["unparseable", "not-a-date"],
    [
      "the epoch, which is how an unset datetime arrives",
      "1970-01-01T00:00:00Z",
    ],
  ])("shows no age when the timestamp is %s", (_case, updatedAt) => {
    render(
      <RunPhase
        status={RUNNING}
        code="running_agent"
        label={null}
        updatedAt={updatedAt}
      />,
    );

    expect(screen.queryByTestId("run-phase-age")).not.toBeInTheDocument();
  });

  it("drops the age on a failed run, whose phase is where it stopped rather than something still running", () => {
    // A months-old failure would otherwise render an absolute date — a
    // second timestamp beside the one the row already shows.
    render(
      <RunPhase
        status={AutomationRunStatus.FAILED}
        code="sandbox_provisioning"
        label={null}
        updatedAt={minutesAgo(41)}
      />,
    );

    expect(screen.getByTestId("run-phase")).toBeInTheDocument();
    expect(screen.queryByTestId("run-phase-age")).not.toBeInTheDocument();
  });
});

describe("RunPhase — reachable without a mouse", () => {
  it("leaves the whole phase, age included, in the accessibility tree", () => {
    // Truncation is CSS only, so a 200-character label is readable in full
    // by a screen reader as long as nothing hides it — and nothing may hide
    // it behind a tab stop, because every surface nests the phase inside a
    // link, where a focusable descendant is invalid and swallows Enter.
    const label = "x".repeat(200);

    render(
      <RunPhase
        status={RUNNING}
        code="poll_prs"
        label={label}
        updatedAt={minutesAgo(7)}
      />,
    );

    const text = screen.getByTestId("run-phase");
    const age = screen.getByTestId("run-phase-age");
    expect(text).toHaveTextContent(label);
    expect(age).toHaveTextContent("il y a 7min");
    expect(text).not.toHaveAttribute("aria-hidden");
    expect(age).not.toHaveAttribute("aria-hidden");
    // No focusable ancestor: the phase must never be its own tab stop
    // inside the link every surface wraps it in.
    expect(text.closest("[tabindex]")).toBeNull();
  });
});
