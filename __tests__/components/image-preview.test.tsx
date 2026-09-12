import { ImagePreview } from "#/components/features/images/image-preview";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

const SRC = "https://example.com/image.jpg";

describe("ImagePreview", () => {
  it("should render an image", () => {
    render(<ImagePreview src={SRC} onRemove={vi.fn} />);

    expect(screen.getByTestId("image-preview")).toBeInTheDocument();
    expect(screen.getByRole("img")).toHaveAttribute("src", SRC);
  });

  it("should call onRemove when the close button is clicked", async () => {
    const user = userEvent.setup();
    const onRemoveMock = vi.fn();
    render(<ImagePreview src={SRC} onRemove={onRemoveMock} />);

    await user.click(
      screen.getByRole("button", { name: "BUTTON$REMOVE_IMAGE" }),
    );

    expect(onRemoveMock).toHaveBeenCalledOnce();
  });

  it("should not display the close button when onRemove is not provided", () => {
    render(<ImagePreview src={SRC} />);

    expect(
      screen.queryByRole("button", { name: "BUTTON$REMOVE_IMAGE" }),
    ).not.toBeInTheDocument();
  });

  it("should not open the lightbox until the thumbnail is clicked", () => {
    render(<ImagePreview src={SRC} />);

    expect(screen.getByTestId("expand-image-button")).toBeInTheDocument();
    expect(screen.queryByTestId("image-lightbox")).not.toBeInTheDocument();
  });

  it("should open a full-size lightbox when the thumbnail is clicked", async () => {
    const user = userEvent.setup();
    render(<ImagePreview src={SRC} />);

    await user.click(screen.getByTestId("expand-image-button"));

    const lightbox = screen.getByTestId("image-lightbox");
    expect(lightbox).toBeInTheDocument();
    // The full-size image renders the same source as the thumbnail — the
    // thumbnail is only a CSS downscale, so no refetch is needed.
    expect(within(lightbox).getByRole("img")).toHaveAttribute("src", SRC);
  });

  it("should keep the lightbox out of the remove flow", async () => {
    const user = userEvent.setup();
    const onRemoveMock = vi.fn();
    render(<ImagePreview src={SRC} onRemove={onRemoveMock} />);

    await user.click(
      screen.getByRole("button", { name: "BUTTON$REMOVE_IMAGE" }),
    );

    expect(onRemoveMock).toHaveBeenCalledOnce();
    expect(screen.queryByTestId("image-lightbox")).not.toBeInTheDocument();
  });

  it("should close the lightbox via its close button", async () => {
    const user = userEvent.setup();
    render(<ImagePreview src={SRC} />);

    await user.click(screen.getByTestId("expand-image-button"));
    await user.click(screen.getByTestId("image-lightbox-close"));

    expect(screen.queryByTestId("image-lightbox")).not.toBeInTheDocument();
  });

  it("should close the lightbox when Escape is pressed", async () => {
    const user = userEvent.setup();
    render(<ImagePreview src={SRC} />);

    await user.click(screen.getByTestId("expand-image-button"));
    await user.keyboard("{Escape}");

    expect(screen.queryByTestId("image-lightbox")).not.toBeInTheDocument();
  });
});
