import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { SectionCard } from "#/components/features/automations/detail/section-card";

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
