import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { UploadedFile } from "#/components/features/chat/uploaded-file";
import { UploadedImage } from "#/components/features/chat/uploaded-image";
import { I18nKey } from "#/i18n/declaration";

vi.mock("react-i18next", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-i18next")>()),
  useTranslation: () => ({
    t: (key: string, options?: { fileName?: string }) =>
      options?.fileName ? `${key} ${options.fileName}` : key,
  }),
}));

describe("attachment remove buttons", () => {
  beforeEach(() => {
    vi.stubGlobal("URL", {
      createObjectURL: (file: File) => `blob:preview/${file.name}`,
      revokeObjectURL: vi.fn(),
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("names the remove button after the attached file it removes", async () => {
    const user = userEvent.setup();
    const onRemove = vi.fn();
    render(
      <UploadedFile
        file={new File(["notes"], "qa-note.txt", { type: "text/plain" })}
        onRemove={onRemove}
      />,
    );

    await user.click(
      screen.getByRole("button", {
        name: `${I18nKey.CHAT_INTERFACE$REMOVE_FILE} qa-note.txt`,
      }),
    );

    expect(onRemove).toHaveBeenCalledTimes(1);
  });

  it("names the remove button of an attached image", async () => {
    const user = userEvent.setup();
    const onRemove = vi.fn();
    render(
      <UploadedImage
        image={new File(["image"], "diagram.png", { type: "image/png" })}
        onRemove={onRemove}
      />,
    );

    await user.click(
      screen.getByRole("button", { name: I18nKey.BUTTON$REMOVE_IMAGE }),
    );

    expect(onRemove).toHaveBeenCalledTimes(1);
  });
});
