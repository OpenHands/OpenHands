import { render, fireEvent, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useFileHandling } from "./use-file-handling";

const TestComponent = ({
  onFilesPaste,
}: {
  onFilesPaste: (files: File[]) => void;
}) => {
  const { fileInputRef, handleFileInputChange } = useFileHandling(onFilesPaste);
  return (
    <input
      type="file"
      multiple
      data-testid="upload-image-input"
      ref={fileInputRef}
      onChange={handleFileInputChange}
    />
  );
};

describe("useFileHandling", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("passes picked files to the upload handler", () => {
    const onFilesPaste = vi.fn();
    render(<TestComponent onFilesPaste={onFilesPaste} />);
    const input = screen.getByTestId("upload-image-input");

    const file = new File(["hello"], "notes.txt", { type: "text/plain" });
    fireEvent.change(input, { target: { files: [file] } });

    expect(onFilesPaste).toHaveBeenCalledTimes(1);
    expect(onFilesPaste).toHaveBeenCalledWith([file], undefined);
  });

  it("resets the input after a pick so the same file can be picked again", () => {
    // A browser file input whose value is unchanged does not fire `change`
    // when the user picks the same file again - e.g. after removing the
    // attachment chip and re-attaching it. The handler must clear the input
    // (value = "") once it has consumed the picked files. jsdom does not
    // model the value->files coupling, so assert the reset call itself.
    const valueSetter = vi.fn();
    const original = Object.getOwnPropertyDescriptor(
      HTMLInputElement.prototype,
      "value",
    );
    vi.spyOn(HTMLInputElement.prototype, "value", "set").mockImplementation(
      function setValue(this: HTMLInputElement, v: string) {
        valueSetter(v);
        original?.set?.call(this, v);
      },
    );

    const onFilesPaste = vi.fn();
    render(<TestComponent onFilesPaste={onFilesPaste} />);
    const input = screen.getByTestId("upload-image-input");

    const file = new File(["hello"], "notes.txt", { type: "text/plain" });
    fireEvent.change(input, { target: { files: [file] } });

    expect(onFilesPaste).toHaveBeenCalledTimes(1);
    expect(valueSetter).toHaveBeenCalledWith("");
  });
});
