import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { MetadataChip } from "#/components/features/automations/metadata-chip";

describe("MetadataChip", () => {
  it("renders the label and icon", () => {
    render(
      <MetadataChip
        icon={<span data-testid="chip-icon" />}
        label="acme/frontend-app"
      />,
    );

    expect(screen.getByText("acme/frontend-app")).toBeInTheDocument();
    expect(screen.getByTestId("chip-icon")).toBeInTheDocument();
  });
});
