import type { ReactNode } from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SaveAsSecretToggle } from "#/components/features/mcp-page/save-as-secret-toggle";

// HeroUI's Tooltip (used inside StyledTooltip) only mounts its content on a
// real hover event, which jsdom doesn't fire. Stub it so the content renders
// eagerly and is queryable in tests.
vi.mock("#/components/shared/buttons/styled-tooltip", () => ({
  StyledTooltip: ({
    content,
    children,
  }: {
    content: ReactNode;
    children: ReactNode;
  }) => (
    <>
      {children}
      <span data-testid="styled-tooltip-content">{content}</span>
    </>
  ),
}));

describe("SaveAsSecretToggle", () => {
  // ── rendering ──────────────────────────────────────────────────────────────

  it("attaches data-testid to the label using fieldKey", () => {
    render(
      <SaveAsSecretToggle fieldKey="MY_KEY" checked={false} onToggle={vi.fn()} />,
    );
    expect(
      screen.getByTestId("mcp-install-save-secret-MY_KEY"),
    ).toBeInTheDocument();
  });

  it("renders the field key inside a <code> element", () => {
    render(
      <SaveAsSecretToggle fieldKey="MY_KEY" checked={false} onToggle={vi.fn()} />,
    );
    const codeEl = screen.getByText("MY_KEY");
    expect(codeEl.tagName.toLowerCase()).toBe("code");
  });

  it("renders the checkbox unchecked when checked=false", () => {
    render(
      <SaveAsSecretToggle fieldKey="KEY" checked={false} onToggle={vi.fn()} />,
    );
    expect(screen.getByRole("checkbox")).not.toBeChecked();
  });

  it("renders the checkbox checked when checked=true", () => {
    render(
      <SaveAsSecretToggle fieldKey="KEY" checked={true} onToggle={vi.fn()} />,
    );
    expect(screen.getByRole("checkbox")).toBeChecked();
  });

  // ── accessibility ──────────────────────────────────────────────────────────

  it("the info button has an aria-label describing its purpose", () => {
    render(
      <SaveAsSecretToggle fieldKey="KEY" checked={false} onToggle={vi.fn()} />,
    );
    // t(key) => key in tests, so aria-label equals the raw i18n key.
    const infoBtn = screen.getByRole("button");
    expect(infoBtn).toHaveAttribute(
      "aria-label",
      "MCP$SAVE_AS_SECRET_TOOLTIP",
    );
  });

  it("the visual track is hidden from the accessibility tree (aria-hidden)", () => {
    const { container } = render(
      <SaveAsSecretToggle fieldKey="KEY" checked={false} onToggle={vi.fn()} />,
    );
    // The decorative <span> that forms the visual slider track.
    const track = container.querySelector("span[aria-hidden='true']");
    expect(track).toBeInTheDocument();
  });

  // ── interaction ────────────────────────────────────────────────────────────

  it("calls onToggle(true) when the unchecked checkbox is clicked", () => {
    const onToggle = vi.fn();
    render(
      <SaveAsSecretToggle fieldKey="KEY" checked={false} onToggle={onToggle} />,
    );
    fireEvent.click(screen.getByRole("checkbox"));
    expect(onToggle).toHaveBeenCalledWith(true);
  });

  it("calls onToggle(false) when the checked checkbox is clicked", () => {
    const onToggle = vi.fn();
    render(
      <SaveAsSecretToggle fieldKey="KEY" checked={true} onToggle={onToggle} />,
    );
    fireEvent.click(screen.getByRole("checkbox"));
    expect(onToggle).toHaveBeenCalledWith(false);
  });

  // ── tooltip ────────────────────────────────────────────────────────────────

  it("passes tooltip text to StyledTooltip as its content prop", () => {
    render(
      <SaveAsSecretToggle fieldKey="KEY" checked={false} onToggle={vi.fn()} />,
    );
    // The mock renders StyledTooltip's content prop into a <span>.
    // t(key) => key, so the rendered text is the raw i18n key.
    expect(screen.getByTestId("styled-tooltip-content")).toHaveTextContent(
      "MCP$SAVE_AS_SECRET_TOOLTIP",
    );
  });
});
