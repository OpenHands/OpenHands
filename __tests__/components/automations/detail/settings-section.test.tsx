import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { SettingsSection } from "#/components/features/automations/detail/settings-section";
import type { Automation } from "#/types/automation";

const automation: Automation = {
  id: "auto-1",
  name: "Daily digest",
  prompt: "Summarize PRs",
  trigger: { type: "cron", schedule: "0 9 * * *" },
  enabled: true,
  created_at: "2024-04-10T12:00:00Z",
  updated_at: "2024-04-10T12:00:00Z",
  timeout: 600,
};

describe("SettingsSection", () => {
  it("shows who it runs as, its timeout and when it was created", () => {
    render(<SettingsSection automation={automation} runsAs="jdoe@acme.com" />);

    expect(screen.getByText("jdoe@acme.com")).toBeInTheDocument();
    expect(screen.getByText("10m")).toBeInTheDocument();
    expect(screen.getByText("Apr 10, 2024")).toBeInTheDocument();
    expect(
      screen.queryByTestId("service-account-badge"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText("AUTOMATIONS$DETAIL$CREATED_BY"),
    ).not.toBeInTheDocument();
  });

  it("badges a service account under Runs As and shows its creator", () => {
    render(
      <SettingsSection
        automation={automation}
        runsAs="ci-bot"
        runsAsServiceAccount
        createdBy="jdoe@acme.com"
      />,
    );

    expect(screen.getByText("ci-bot")).toBeInTheDocument();
    expect(screen.getByTestId("service-account-badge")).toHaveAttribute(
      "title",
      "AUTOMATIONS$SERVICE_ACCOUNT_LABEL",
    );
    expect(
      screen.getByText("AUTOMATIONS$DETAIL$CREATED_BY"),
    ).toBeInTheDocument();
    expect(screen.getByText("jdoe@acme.com")).toBeInTheDocument();
  });

  it("leaves out runs-as and timeout when there is nothing to show", () => {
    // Local backends and in-flight lookups pass no identity.
    render(<SettingsSection automation={{ ...automation, timeout: null }} />);

    expect(
      screen.queryByText("AUTOMATIONS$DETAIL$RUNS_AS"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText("AUTOMATIONS$DETAIL$TIMEOUT"),
    ).not.toBeInTheDocument();
  });
});
