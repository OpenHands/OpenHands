import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderToString } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { UploadedImage } from "#/components/features/chat/uploaded-image";

vi.mock("lucide-react", () => ({
  LoaderCircle: ({
    className,
    color,
  }: {
    className?: string;
    color?: string;
  }) => (
    <div data-testid="image-loader" data-color={color} className={className} />
  ),
}));

vi.mock("#/components/features/chat/remove-file-button", () => ({
  RemoveFileButton: ({ onClick }: { onClick: () => void }) => (
    <button type="button" data-testid="remove-image" onClick={onClick}>
      Remove image
    </button>
  ),
}));

vi.mock(
  "#/components/features/chat/pasted-image-upload-as-file-button",
  () => ({
    PastedImageUploadAsFileButton: ({
      active,
      onToggle,
    }: {
      active: boolean;
      onToggle: () => void;
    }) => (
      <button
        type="button"
        data-testid="upload-image-as-file"
        aria-pressed={active}
        onClick={onToggle}
      >
        Toggle upload as file
      </button>
    ),
  }),
);

function createImage(name = "diagram.png"): File {
  return new File(["image bytes"], name, { type: "image/png" });
}

function installObjectUrlMocks() {
  const createObjectURL = vi.fn((file: File) => `blob:preview/${file.name}`);
  const revokeObjectURL = vi.fn();

  vi.stubGlobal("URL", {
    createObjectURL,
    revokeObjectURL,
  });

  return { createObjectURL, revokeObjectURL };
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("uploaded image preview", () => {
  it("shows the image and supports removal with default options", async () => {
    const user = userEvent.setup();
    const { createObjectURL, revokeObjectURL } = installObjectUrlMocks();
    const image = createImage();
    const onRemove = vi.fn();
    const { unmount } = render(
      <UploadedImage
        image={image}
        onRemove={onRemove}
        onToggleUploadAsFile={vi.fn()}
      />,
    );

    const preview = screen.getByRole("img", { name: "diagram.png" });
    expect(preview).toHaveAttribute("src", "blob:preview/diagram.png");
    expect(createObjectURL).toHaveBeenCalledWith(image);
    expect(screen.queryByTestId("image-loader")).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("upload-image-as-file"),
    ).not.toBeInTheDocument();

    await user.click(screen.getByTestId("remove-image"));
    expect(onRemove).toHaveBeenCalledOnce();

    unmount();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:preview/diagram.png");
  });

  it("renders no image before an object URL is available", () => {
    const initialMarkup = renderToString(
      <UploadedImage image={createImage()} onRemove={vi.fn()} />,
    );

    expect(initialMarkup).not.toContain("<img");
  });

  it("starts the upload-as-file control inactive when it is enabled", () => {
    installObjectUrlMocks();

    render(
      <UploadedImage
        image={createImage()}
        onRemove={vi.fn()}
        showUploadAsFileToggle
        onToggleUploadAsFile={vi.fn()}
      />,
    );

    expect(screen.getByTestId("upload-image-as-file")).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });

  it("shows a loader and forwards an active upload-as-file toggle", async () => {
    const user = userEvent.setup();
    const { revokeObjectURL } = installObjectUrlMocks();
    const onToggleUploadAsFile = vi.fn();
    const { unmount } = render(
      <UploadedImage
        image={createImage("large-image.png")}
        onRemove={vi.fn()}
        isLoading
        showUploadAsFileToggle
        uploadAsFileActive
        onToggleUploadAsFile={onToggleUploadAsFile}
      />,
    );

    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getByTestId("image-loader")).toHaveClass("animate-spin");
    expect(screen.getByTestId("image-loader")).toHaveAttribute(
      "data-color",
      "white",
    );
    const toggle = screen.getByTestId("upload-image-as-file");
    expect(toggle).toHaveAttribute("aria-pressed", "true");

    await user.click(toggle);
    expect(onToggleUploadAsFile).toHaveBeenCalledOnce();

    unmount();
    expect(revokeObjectURL).toHaveBeenCalledWith(
      "blob:preview/large-image.png",
    );
  });

  it("refreshes the object URL for a new image and hides an unusable toggle", () => {
    const { createObjectURL, revokeObjectURL } = installObjectUrlMocks();
    const firstImage = createImage("first.png");
    const secondImage = createImage("second.png");
    const onRemove = vi.fn();
    const { rerender, unmount } = render(
      <UploadedImage
        image={firstImage}
        onRemove={onRemove}
        showUploadAsFileToggle
      />,
    );

    expect(screen.getByRole("img", { name: "first.png" })).toHaveAttribute(
      "src",
      "blob:preview/first.png",
    );
    expect(
      screen.queryByTestId("upload-image-as-file"),
    ).not.toBeInTheDocument();

    rerender(
      <UploadedImage
        image={secondImage}
        onRemove={onRemove}
        showUploadAsFileToggle
      />,
    );

    expect(screen.getByRole("img", { name: "second.png" })).toHaveAttribute(
      "src",
      "blob:preview/second.png",
    );
    expect(createObjectURL).toHaveBeenNthCalledWith(1, firstImage);
    expect(createObjectURL).toHaveBeenNthCalledWith(2, secondImage);
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:preview/first.png");

    unmount();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:preview/second.png");
  });
});
