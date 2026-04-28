import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { FeatureList } from "#/components/features/onboarding/feature-list";

describe("FeatureList", () => {
  it("should render a list of features", () => {
    const features = ["Feature 1", "Feature 2", "Feature 3"];
    render(<FeatureList features={features} />);

    expect(screen.getByText("Feature 1")).toBeInTheDocument();
    expect(screen.getByText("Feature 2")).toBeInTheDocument();
    expect(screen.getByText("Feature 3")).toBeInTheDocument();
  });

  it("should render bullet points for each feature", () => {
    const features = ["Feature 1", "Feature 2"];
    render(<FeatureList features={features} />);

    const bullets = screen.getAllByText("•");
    expect(bullets).toHaveLength(2);
  });

  it("should render an empty list when no features provided", () => {
    render(<FeatureList features={[]} />);

    const list = screen.getByRole("list");
    expect(list).toBeInTheDocument();
    expect(list.children).toHaveLength(0);
  });

  it("should render each feature as a list item", () => {
    const features = ["Feature 1", "Feature 2"];
    render(<FeatureList features={features} />);

    const listItems = screen.getAllByRole("listitem");
    expect(listItems).toHaveLength(2);
  });
});
