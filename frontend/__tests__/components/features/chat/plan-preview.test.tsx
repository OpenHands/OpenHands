import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { PlanPreview } from "#/components/features/chat/plan-preview";

// Mock the feature flag to always return true (not testing feature flag behavior)
vi.mock("#/utils/feature-flags", () => ({
  USE_PLANNING_AGENT: vi.fn(() => true),
}));

// Mock i18n - need to preserve initReactI18next and I18nextProvider for test-utils
vi.mock("react-i18next", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-i18next")>();
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string) => key,
    }),
  };
});

describe("PlanPreview", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render nothing when planContent is null", () => {
    renderWithProviders(<PlanPreview planContent={null} />);

    const contentDiv = screen.getByTestId("plan-preview-content");
    expect(contentDiv).toBeInTheDocument();
    expect(contentDiv.textContent?.trim() || "").toBe("");
  });

  it("should render nothing when planContent is undefined", () => {
    renderWithProviders(<PlanPreview planContent={undefined} />);

    const contentDiv = screen.getByTestId("plan-preview-content");
    expect(contentDiv).toBeInTheDocument();
    expect(contentDiv.textContent?.trim() || "").toBe("");
  });

  it("should render markdown content when planContent is provided", () => {
    const planContent = "# Plan Title\n\nThis is the plan content.";

    const { container } = renderWithProviders(
      <PlanPreview planContent={planContent} />,
    );

    // Check that component rendered and contains the content (markdown may break up text)
    expect(container.firstChild).not.toBeNull();
    expect(container.textContent).toContain("Plan Title");
    expect(container.textContent).toContain("This is the plan content.");
  });

  it("should render full content when length is less than or equal to 300 characters", () => {
    const planContent = "A".repeat(300);

    const { container } = renderWithProviders(
      <PlanPreview planContent={planContent} />,
    );

    // Content should be present (may be broken up by markdown)
    expect(container.textContent).toContain(planContent);
    expect(screen.queryByText(/COMMON\$READ_MORE/i)).not.toBeInTheDocument();
  });

  it("should truncate content when length exceeds 300 characters", () => {
    const longContent = "A".repeat(350);

    const { container } = renderWithProviders(
      <PlanPreview planContent={longContent} />,
    );

    // Truncated content should be present (may be broken up by markdown)
    expect(container.textContent).toContain("A".repeat(300));
    expect(container.textContent).toContain("...");
    expect(container.textContent).toContain("COMMON$READ_MORE");
  });

  it("should call onViewClick when View button is clicked", async () => {
    const user = userEvent.setup();
    const onViewClick = vi.fn();

    renderWithProviders(
      <PlanPreview planContent="Plan content" onViewClick={onViewClick} />,
    );

    const viewButton = screen.getByTestId("plan-preview-view-button");
    expect(viewButton).toBeInTheDocument();

    await user.click(viewButton);

    expect(onViewClick).toHaveBeenCalledTimes(1);
  });

  it("should call onViewClick when Read More button is clicked", async () => {
    const user = userEvent.setup();
    const onViewClick = vi.fn();
    const longContent = "A".repeat(350);

    renderWithProviders(
      <PlanPreview planContent={longContent} onViewClick={onViewClick} />,
    );

    const readMoreButton = screen.getByTestId("plan-preview-read-more-button");
    expect(readMoreButton).toBeInTheDocument();

    await user.click(readMoreButton);

    expect(onViewClick).toHaveBeenCalledTimes(1);
  });

  it("should call onBuildClick when Build button is clicked", async () => {
    const user = userEvent.setup();
    const onBuildClick = vi.fn();

    renderWithProviders(
      <PlanPreview planContent="Plan content" onBuildClick={onBuildClick} />,
    );

    const buildButton = screen.getByTestId("plan-preview-build-button");
    expect(buildButton).toBeInTheDocument();

    await user.click(buildButton);

    expect(onBuildClick).toHaveBeenCalledTimes(1);
  });

  it("should render header with PLAN_MD text", () => {
    const { container } = renderWithProviders(
      <PlanPreview planContent="Plan content" />,
    );

    // Check that the translation key is rendered (i18n mock returns the key)
    expect(container.textContent).toContain("COMMON$PLAN_MD");
  });

  it("should render plan content", () => {
    const planContent = `# Heading 1
## Heading 2
- List item 1
- List item 2

**Bold text** and *italic text*`;

    const { container } = renderWithProviders(
      <PlanPreview planContent={planContent} />,
    );

    expect(container.textContent).toContain("Heading 1");
    expect(container.textContent).toContain("Heading 2");
  });
});
