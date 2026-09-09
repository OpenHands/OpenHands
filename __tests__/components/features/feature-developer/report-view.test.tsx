import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ReportView } from "#/components/features/feature-developer/report-view";
import { I18nKey } from "#/i18n/declaration";

const MARKDOWN = `| Title | Status | Cost | Branch |
| --- | --- | --- | --- |
| Add login form | passed | $0.2500 | feat/login-form |
`;

describe("ReportView", () => {
  it("renders the markdown summary table", () => {
    render(<ReportView markdown={MARKDOWN} />);

    expect(screen.getByTestId("feature-dev-report")).toHaveTextContent(
      I18nKey.FEATURE_DEV$REPORT,
    );
    expect(screen.getByTestId("feature-dev-report")).toHaveTextContent(
      "Add login form",
    );
    expect(screen.getByTestId("feature-dev-report")).toHaveTextContent(
      "feat/login-form",
    );
  });
});
