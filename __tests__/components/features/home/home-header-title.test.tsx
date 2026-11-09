import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { renderWithProviders } from "test-utils";
import { HomeHeaderTitle } from "#/components/features/home/home-header/home-header-title";

describe("HomeHeaderTitle", () => {
  it("frames the splash copy around engineering work, not greenfield building", () => {
    renderWithProviders(<HomeHeaderTitle />);

    expect(screen.getByText("HOME$WHAT_TO_WORK_ON")).toBeInTheDocument();
    expect(screen.getByTestId("home-header-subtitle")).toHaveTextContent(
      "HOME$ENGINEERING_TASKS_SUBHEADER",
    );
  });
});
