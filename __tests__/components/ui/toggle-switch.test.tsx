import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ToggleSwitchVisual } from "#/ui/toggle-switch";

describe("ToggleSwitchVisual RTL and LTR appearance", () => {
  it("renders md size with correct LTR and RTL translation classes when unchecked", () => {
    const { container } = render(
      <ToggleSwitchVisual enabled={false} size="md" />,
    );
    const track = container.firstElementChild as HTMLElement;
    const thumb = track.firstElementChild as HTMLElement;

    expect(track).toHaveClass("h-5.5", "w-10", "border");
    expect(thumb).toHaveClass(
      "size-4",
      "translate-x-0.5",
      "rtl:-translate-x-0.5",
    );
  });

  it("renders md size with correct LTR and RTL translation classes when checked", () => {
    const { container } = render(
      <ToggleSwitchVisual enabled={true} size="md" />,
    );
    const track = container.firstElementChild as HTMLElement;
    const thumb = track.firstElementChild as HTMLElement;

    expect(track).toHaveClass("h-5.5", "w-10", "border");
    expect(thumb).toHaveClass(
      "size-4",
      "translate-x-5.25",
      "rtl:-translate-x-5.25",
    );
  });

  it("renders sm size with correct LTR and RTL translation classes when unchecked", () => {
    const { container } = render(
      <ToggleSwitchVisual enabled={false} size="sm" />,
    );
    const track = container.firstElementChild as HTMLElement;
    const thumb = track.firstElementChild as HTMLElement;

    expect(track).toHaveClass("h-3.5", "w-6", "p-0.75");
    expect(thumb).toHaveClass("size-2", "translate-x-0", "rtl:translate-x-0");
  });

  it("renders sm size with correct LTR and RTL translation classes when checked", () => {
    const { container } = render(
      <ToggleSwitchVisual enabled={true} size="sm" />,
    );
    const track = container.firstElementChild as HTMLElement;
    const thumb = track.firstElementChild as HTMLElement;

    expect(track).toHaveClass("h-3.5", "w-6", "p-0.75");
    expect(thumb).toHaveClass(
      "size-2",
      "translate-x-2.5",
      "rtl:-translate-x-2.5",
    );
  });

  it("mirrors thumb position under RTL container for both sizes and states", () => {
    const { container } = render(
      <div dir="rtl">
        <ToggleSwitchVisual enabled={false} size="md" />
        <ToggleSwitchVisual enabled={true} size="md" />
        <ToggleSwitchVisual enabled={false} size="sm" />
        <ToggleSwitchVisual enabled={true} size="sm" />
      </div>,
    );

    const tracks = container.querySelectorAll("span[aria-hidden='true']");
    expect(tracks).toHaveLength(4);

    // md unchecked
    expect(tracks[0].firstElementChild).toHaveClass("rtl:-translate-x-0.5");
    // md checked
    expect(tracks[1].firstElementChild).toHaveClass("rtl:-translate-x-5.25");
    // sm unchecked
    expect(tracks[2].firstElementChild).toHaveClass("rtl:translate-x-0");
    // sm checked
    expect(tracks[3].firstElementChild).toHaveClass("rtl:-translate-x-2.5");
  });
});
