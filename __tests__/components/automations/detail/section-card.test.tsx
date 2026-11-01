import { render, screen } from "@testing-library/react";
import { afterEach, describe, it, expect, vi } from "vitest";
import { SectionCard } from "#/components/features/automations/detail/section-card";
import { ActivitySection } from "#/components/features/automations/detail/activity-section";
import { I18nKey } from "#/i18n/declaration";

const mocks = vi.hoisted(() => ({
  useTranslation: vi.fn(),
}));

vi.mock("react-i18next", () => ({
  useTranslation: (namespace?: unknown) => {
    mocks.useTranslation(namespace);
    return {
      t: (key: string, options?: { count?: number }) =>
        options?.count === undefined ? key : `${key}:${options.count}`,
      i18n: { language: "en-US" },
    };
  },
}));

describe("SectionCard", () => {
  it("renders the title and children content", () => {
    render(
      <SectionCard icon={<span data-testid="icon" />} title="Test Section">
        <p>{String("Section content")}</p>
      </SectionCard>,
    );

    expect(screen.getByText("Test Section")).toBeInTheDocument();
    expect(screen.getByText("Section content")).toBeInTheDocument();
    expect(screen.getByTestId("icon")).toBeInTheDocument();
  });
});

describe("Automation activity summary", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("shows the creation date and reports an automation that has never run", () => {
    render(
      <ActivitySection createdAt="2024-04-10T12:00:00Z" lastRunAt={null} />,
    );

    expect(mocks.useTranslation).toHaveBeenCalledWith("openhands");
    expect(
      screen.getByText(I18nKey.AUTOMATIONS$DETAIL$ACTIVITY),
    ).toBeInTheDocument();
    expect(
      screen.getByText(I18nKey.AUTOMATIONS$DETAIL$CREATED),
    ).toBeInTheDocument();
    expect(screen.getByText("Apr 10, 2024")).toBeInTheDocument();
    expect(
      screen.getByText(I18nKey.AUTOMATIONS$DETAIL$TIME_NEVER),
    ).toBeInTheDocument();
  });

  it.each([
    {
      description: "less than a minute ago",
      lastRunAt: "2026-06-15T11:59:30Z",
      expected: I18nKey.AUTOMATIONS$DETAIL$TIME_JUST_NOW,
    },
    {
      description: "minutes ago",
      lastRunAt: "2026-06-15T11:23:00Z",
      expected: `${I18nKey.AUTOMATIONS$DETAIL$TIME_MINUTES_AGO}:37`,
    },
    {
      description: "exactly one minute ago",
      lastRunAt: "2026-06-15T11:59:00Z",
      expected: `${I18nKey.AUTOMATIONS$DETAIL$TIME_MINUTES_AGO}:1`,
    },
    {
      description: "hours ago",
      lastRunAt: "2026-06-15T06:30:00Z",
      expected: `${I18nKey.AUTOMATIONS$DETAIL$TIME_HOURS_AGO}:5`,
    },
    {
      description: "exactly one hour ago",
      lastRunAt: "2026-06-15T11:00:00Z",
      expected: `${I18nKey.AUTOMATIONS$DETAIL$TIME_HOURS_AGO}:1`,
    },
    {
      description: "yesterday",
      lastRunAt: "2026-06-14T10:00:00Z",
      expected: I18nKey.AUTOMATIONS$DETAIL$TIME_YESTERDAY,
    },
    {
      description: "exactly one day ago",
      lastRunAt: "2026-06-14T12:00:00Z",
      expected: I18nKey.AUTOMATIONS$DETAIL$TIME_YESTERDAY,
    },
    {
      description: "several days ago",
      lastRunAt: "2026-06-12T07:00:00Z",
      expected: `${I18nKey.AUTOMATIONS$DETAIL$TIME_DAYS_AGO}:3`,
    },
    {
      description: "exactly one week ago",
      lastRunAt: "2026-06-08T12:00:00Z",
      expected: "Jun 8, 2026",
    },
  ])("shows a run from $description", ({ lastRunAt, expected }) => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-06-15T12:00:00Z"));

    render(
      <ActivitySection
        createdAt="2024-04-10T12:00:00Z"
        lastRunAt={lastRunAt}
      />,
    );

    expect(screen.getByText(expected)).toBeInTheDocument();
  });

  it("uses an absolute localized date for a run at least a week old", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-06-15T12:00:00Z"));

    render(
      <ActivitySection
        createdAt="2024-04-10T12:00:00Z"
        lastRunAt="2020-01-02T12:00:00Z"
      />,
    );

    expect(screen.getByText("Jan 2, 2020")).toBeInTheDocument();
  });
});
