import { ImagePreview } from "#/components/features/images/image-preview";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

describe("ImagePreview", () => {
  it("should render an image", () => {
    render(
      <ImagePreview src="https://example.com/image.jpg" onRemove={vi.fn} />,
    );
    const img = screen.getByRole("img");

    expect(screen.getByTestId("image-preview")).toBeInTheDocument();
    expect(img).toHaveAttribute("src", "https://example.com/image.jpg");
  });

  it("should call onRemove when the close button is clicked", async () => {
    const user = userEvent.setup();
    const onRemoveMock = vi.fn();
    render(
      <ImagePreview
        src="https://example.com/image.jpg"
        onRemove={onRemoveMock}
      />,
    );

    const closeButton = screen.getByRole("button");
    await user.click(closeButton);

    expect(onRemoveMock).toHaveBeenCalledOnce();
  });

  it("should not display the close button when onRemove is not provided", () => {
    render(<ImagePreview src="https://example.com/image.jpg" />);
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });
});
